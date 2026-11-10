#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cub import CubDataset, cub_collate
from src.utils.amp import autocast_context, make_scaler
from src.utils.ckpt import load_checkpoint, save_checkpoint
from src.utils.config import count_params, load_yaml
from src.utils.device import device_name, get_device
from src.utils.ema import EMA
from src.utils.seed import set_seed
from src.vae.loss import VAELoss
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="runs/vae")
    p.add_argument(
        "--resume",
        default="auto",
        help="Checkpoint path, 'auto' (= out_dir/vae_last.pt if present), or 'none' for fresh start.",
    )
    return p.parse_args()


def resolve_resume(resume: str, out_dir: Path) -> Path | None:
    r = (resume or "auto").strip().lower()
    if r in {"", "none", "no", "false", "0"}:
        return None
    if r in {"auto", "last"}:
        cand = out_dir / "vae_last.pt"
        return cand if cand.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"--resume not found: {path}")
    return path


def build_payload(model, ema, opt, scaler, step, cfg) -> dict:
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": cfg,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} ({device_name(device)})")

    ds = CubDataset(
        args.data_root,
        split="train",
        image_size=int(cfg["image_size"]),
        bbox_expand=float(cfg.get("bbox_expand", 1.1)),
        require_captions=False,
    )
    print(f"CUB train images: {len(ds)} (caption files skipped={ds.skipped_captions})")
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=cub_collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = AutoencoderKL(
        latent_channels=int(cfg.get("latent_channels", 4)),
        channels=tuple(cfg.get("channels", [128, 256, 512, 512])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
    ).to(device)
    print(f"VAE params: {count_params(model)/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9999)))
    scaler = make_scaler(device, cfg.get("precision", "amp_fp16"))
    loss_fn = VAELoss(
        kl_weight=float(cfg.get("kl_weight", 1e-6)),
        lpips_weight=float(cfg.get("lpips_weight", 0.1)),
        gan_weight=float(cfg.get("gan_weight", 0.0)),
    ).to(device)
    writer = SummaryWriter(out_dir / "tb")

    step = 0
    resume_path = resolve_resume(args.resume, out_dir)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt.get("step", 0))
        print(f"Resumed from {resume_path} at step={step}")
    else:
        print("Starting fresh (no resume checkpoint).")

    max_steps = int(cfg["max_steps"])
    accum = max(1, int(cfg.get("grad_accum", 1)))
    save_every = int(cfg.get("save_every", 5000))
    opt.zero_grad(set_to_none=True)
    model.train()
    pbar = tqdm(total=max_steps, initial=step, desc="vae")

    def save_resume(tag: str | None = None) -> None:
        payload = build_payload(model, ema, opt, scaler, step, cfg)
        save_checkpoint(out_dir / "vae_last.pt", payload)
        save_checkpoint(
            out_dir / "vae_ema.pt",
            {
                "model": ema.shadow.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config": cfg,
                **({"scaler": scaler.state_dict()} if scaler is not None else {}),
            },
        )
        if tag:
            save_checkpoint(out_dir / f"vae_step{tag}.pt", payload)
        print(f"Saved resume checkpoint at step={step} -> {out_dir / 'vae_last.pt'}")

    try:
        while step < max_steps:
            for batch in loader:
                x = batch["image"].to(device)
                with autocast_context(device, cfg.get("precision", "amp_fp16")):
                    out = model(x)
                    losses = loss_fn(out, x)
                    loss = losses["loss"] / accum
                if scaler is None:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    if scaler is None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    else:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt)
                        scaler.update()
                    opt.zero_grad(set_to_none=True)
                    ema.update(model)
                if step % int(cfg.get("log_every", 50)) == 0:
                    writer.add_scalar("loss/total", losses["loss"].item(), step)
                    writer.add_scalar("loss/l1", losses["l1"].item(), step)
                    writer.add_scalar("loss/kl", losses["kl"].item(), step)
                    writer.add_scalar("loss/lpips", losses["lpips"].item(), step)
                    pbar.set_postfix(l1=f"{losses['l1'].item():.3f}", kl=f"{losses['kl'].item():.4f}")
                if step > 0 and step % int(cfg.get("sample_every", 2000)) == 0:
                    vis = torch.cat([x[:4], out["recon"][:4].clamp(-1, 1)], dim=0)
                    writer.add_images("recon", (vis + 1) / 2, step)
                step += 1
                pbar.update(1)
                if step > 0 and step % save_every == 0:
                    save_resume(tag=str(step))
                if step >= max_steps:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted — writing resume checkpoint before exit...")
        save_resume(tag=f"interrupt_{step}")
        pbar.close()
        writer.close()
        print("Safe to copy runs/vae/ and continue later with --resume auto")
        return

    pbar.close()
    save_resume(tag=str(step))
    print("Saved", out_dir / "vae_ema.pt")
    print("Next: python scripts/eval_vae.py --ckpt runs/vae/vae_ema.pt ...  (writes latent σ)")


if __name__ == "__main__":
    main()
