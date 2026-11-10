#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cub import CubDataset, cub_collate
from src.data.transforms import minus1_1_to_pil
from src.utils.ckpt import load_checkpoint, save_checkpoint
from src.utils.config import load_yaml
from src.utils.device import get_device
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="runs/vae")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def build_vae(cfg: dict, ckpt: dict, device: torch.device) -> AutoencoderKL:
    model = AutoencoderKL(
        latent_channels=int(cfg.get("latent_channels", 4)),
        channels=tuple(cfg.get("channels", [128, 256, 512, 512])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
    )
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model = build_vae(cfg, ckpt, device)

    train_ds = CubDataset(args.data_root, split="train", image_size=int(cfg["image_size"]), bbox_expand=float(cfg.get("bbox_expand", 1.1)), require_captions=False)
    test_ds = CubDataset(args.data_root, split="test", image_size=int(cfg["image_size"]), bbox_expand=float(cfg.get("bbox_expand", 1.1)), require_captions=False)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=cub_collate)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=cub_collate)

    means = []
    for batch in tqdm(train_loader, desc="sigma"):
        x = batch["image"].to(device)
        mean, _ = model.encode(x)
        means.append(mean.float().cpu())
    all_mean = torch.cat(means, 0)
    sigma = float(all_mean.std())
    print(f"latent σ (train μ std) = {sigma:.6f}")

    psnr_m = PeakSignalNoiseRatio(data_range=2.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_vals = []
    try:
        import lpips

        lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as exc:  # noqa: BLE001
        lpips_net = None
        print("LPIPS unavailable:", exc)

    saved = False
    for batch in tqdm(test_loader, desc="recon"):
        x = batch["image"].to(device)
        mean, _ = model.encode(x)
        recon = model.decode(mean).clamp(-1, 1)
        psnr_m.update(recon, x)
        ssim_m.update(recon, x)
        if lpips_net is not None:
            lpips_vals.append(lpips_net(recon, x).mean().item())
        if not saved:
            grid = []
            for i in range(min(8, x.size(0))):
                grid.append(minus1_1_to_pil(x[i]))
                grid.append(minus1_1_to_pil(recon[i]))
            w = grid[0].width
            h = grid[0].height
            canvas = grid[0].crop((0, 0, w * 2, h * 4)).copy()
            from PIL import Image

            canvas = Image.new("RGB", (w * 2, h * 4))
            for i, im in enumerate(grid[:8]):
                canvas.paste(im, ((i % 2) * w, (i // 2) * h))
            canvas.save(out_dir / "vae_recon_grid.png")
            saved = True

    psnr = float(psnr_m.compute())
    ssim = float(ssim_m.compute())
    lp = float(sum(lpips_vals) / max(len(lpips_vals), 1)) if lpips_vals else float("nan")
    metrics = {"sigma": sigma, "psnr": psnr, "ssim": ssim, "lpips": lp, "n_train": len(train_ds), "n_test": len(test_ds)}
    (out_dir / "latent_scale.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ckpt["sigma"] = sigma
    ckpt["latent_scale"] = sigma
    ckpt["vae_metrics"] = metrics
    save_checkpoint(args.ckpt, ckpt)
    print(json.dumps(metrics, indent=2))
    min_psnr = float(cfg.get("min_psnr", 24.0))
    max_lpips = float(cfg.get("max_lpips", 0.25))
    fail = psnr < min_psnr or (lpips_vals and lp > max_lpips)
    if fail:
        msg = f"VAE gate warning: PSNR {psnr:.2f} (min {min_psnr}) LPIPS {lp:.3f} (max {max_lpips}). Inspect eyes/beaks in {out_dir / 'vae_recon_grid.png'}"
        print(msg)
        if args.strict:
            raise SystemExit(1)
    else:
        print("VAE reconstruction gate passed (warn-only thresholds).")


if __name__ == "__main__":
    main()
