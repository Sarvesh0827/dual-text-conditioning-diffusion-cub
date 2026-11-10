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

from src.data.cached import CachedLatentClipDataset, cached_collate
from src.diffusion.forward import q_sample
from src.diffusion.schedule import CosineSchedule
from src.text.condition import TextCondition
from src.unet.model import count_unet_params, unet_from_config
from src.utils.amp import autocast_context, make_scaler
from src.utils.ckpt import load_checkpoint, save_checkpoint
from src.utils.config import latent_scale_from_ckpt, load_yaml
from src.utils.device import device_name, get_device
from src.utils.ema import EMA
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--latent_cache", default="cache/latents")
    p.add_argument("--clip_cache", default="cache/clip")
    p.add_argument("--out_dir", default="runs/unet_C")
    p.add_argument("--split", default="train")
    p.add_argument(
        "--resume",
        default="auto",
        help="Checkpoint path, 'auto' (= out_dir/unet_last.pt if present), or 'none' for fresh start.",
    )
    return p.parse_args()


def resolve_resume(resume: str, out_dir: Path) -> Path | None:
    r = (resume or "auto").strip().lower()
    if r in {"", "none", "no", "false", "0"}:
        return None
    if r in {"auto", "last"}:
        cand = out_dir / "unet_last.pt"
        return cand if cand.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise FileNotFoundError(f"--resume not found: {path}")
    return path


def build_payload(model, ema, opt, scaler, step, cfg, n_params) -> dict:
    payload = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "config": cfg,
        "n_params": n_params,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if cfg.get("use_part_loss"):
        raise NotImplementedError("use_part_loss is a v1 stub. Keep it false.")
    set_seed(int(cfg.get("seed", 0)))
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} ({device_name(device)}) experiment={cfg.get('experiment', '?')}")
    print(f"use_global_text={cfg['use_global_text']} use_local_text={cfg['use_local_text']}")

    latent_path = Path(args.latent_cache) / f"{args.split}.pt"
    clip_path = Path(args.clip_cache) / f"{args.split}.pt"
    if not latent_path.is_file() or not clip_path.is_file():
        raise FileNotFoundError(
            f"Need {latent_path} and {clip_path}. "
            "Run scripts/cache_latents.py and scripts/cache_clip.py on the 4090 after eval_vae.py."
        )
    ds = CachedLatentClipDataset(latent_path, clip_path)
    vae_ckpt = load_checkpoint(args.vae_ckpt, map_location="cpu")
    sigma = latent_scale_from_ckpt(vae_ckpt)
    if abs(sigma - ds.sigma) > 1e-5:
        raise ValueError(f"VAE σ={sigma} != latent cache σ={ds.sigma}. Recache latents with this VAE.")
    print(f"cached samples={len(ds)} latent_sigma={ds.sigma:.6f}")
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        collate_fn=cached_collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = unet_from_config(cfg).to(device)
    n_params = count_unet_params(model)
    print(f"UNet params: {n_params/1e6:.2f}M (A/B/C must match)")
    if not (80e6 <= n_params <= 200e6):
        print("Warning: param count outside the planned 80–150M band (still OK if A/B/C match).")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.9999)))
    scaler = make_scaler(device, cfg.get("precision", "amp_fp16"))
    schedule = CosineSchedule(int(cfg["timesteps"]), device=device)
    writer = SummaryWriter(out_dir / "tb")
    p_uncond = float(cfg.get("p_uncond", 0.10))
    accum = max(1, int(cfg.get("grad_accum", 1)))
    max_steps = int(cfg["max_steps"])
    save_every = int(cfg.get("save_every", 5000))
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

    opt.zero_grad(set_to_none=True)
    model.train()
    pbar = tqdm(total=max_steps, initial=step, desc=f"unet-{cfg.get('experiment', '')}")

    def save_resume(tag: str | None = None) -> None:
        payload = build_payload(model, ema, opt, scaler, step, cfg, n_params)
        save_checkpoint(out_dir / "unet_last.pt", payload)
        save_checkpoint(
            out_dir / "unet_ema.pt",
            {
                "model": ema.shadow.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config": cfg,
                "n_params": n_params,
                **({"scaler": scaler.state_dict()} if scaler is not None else {}),
            },
        )
        if tag:
            save_checkpoint(out_dir / f"unet_step{tag}.pt", payload)
        print(f"Saved resume checkpoint at step={step} -> {out_dir / 'unet_last.pt'}")

    try:
        while step < max_steps:
            for batch in loader:
                z = batch["z"].to(device)
                cond = TextCondition(tokens=batch["tokens"].to(device), pooled=batch["pooled"].to(device))
                t = torch.randint(0, schedule.timesteps, (z.size(0),), device=device)
                drop = torch.rand(z.size(0), device=device) < p_uncond
                with autocast_context(device, cfg.get("precision", "amp_fp16")):
                    zt, noise = q_sample(z, t, schedule)
                    pred = model(zt, t, cond, drop_cond=drop)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float()) / accum
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
                if step % int(cfg.get("log_every", 100)) == 0:
                    writer.add_scalar("loss/mse", loss.item() * accum, step)
                    pbar.set_postfix(mse=f"{loss.item() * accum:.4f}")
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
        print("Safe to copy this out_dir and continue later with --resume auto")
        return

    pbar.close()
    save_resume(tag=str(step))
    print("Saved", out_dir / "unet_ema.pt")


if __name__ == "__main__":
    main()
