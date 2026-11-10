#!/usr/bin/env python3
"""Short CUDA memory / it/s probe. Run on the 4090 before setting max_steps."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diffusion.forward import q_sample
from src.diffusion.schedule import CosineSchedule
from src.text.condition import TextCondition
from src.unet.model import count_unet_params, unet_from_config
from src.utils.amp import autocast_context
from src.utils.config import load_yaml
from src.utils.device import device_name, get_device
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unet_C.yaml")
    p.add_argument("--data_root", default="")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()
    print(f"device={device} ({device_name(device)})")
    if device.type != "cuda":
        print("This benchmark is for the RTX 4090. On CPU it only checks that the graph builds.")
    bs = int(args.batch_size or cfg.get("batch_size", 8))
    vae = AutoencoderKL().to(device)
    unet = unet_from_config(cfg).to(device)
    print(f"VAE params {count_unet_params(vae)/1e6:.2f}M  UNet params {count_unet_params(unet)/1e6:.2f}M")
    schedule = CosineSchedule(int(cfg.get("timesteps", 1000)), device=device)
    x = torch.randn(bs, 3, 256, 256, device=device)
    z = torch.randn(bs, 4, 64, 64, device=device)
    tokens = torch.randn(bs, 77, 512, device=device)
    pooled = torch.randn(bs, 512, device=device)
    cond = TextCondition(tokens=tokens, pooled=pooled)
    t = torch.zeros(bs, device=device, dtype=torch.long)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    # VAE step
    t0 = time.time()
    with autocast_context(device, cfg.get("precision", "amp_fp16")):
        rec = vae(x)
        vloss = torch.nn.functional.l1_loss(rec["recon"], x)
    vloss.backward()
    opt_v = torch.optim.AdamW(vae.parameters(), lr=1e-4)
    opt_v.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    vae_s = time.time() - t0

    # UNet steps
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.get("precision", "amp_fp16")):
            zt, noise = q_sample(z, t, schedule)
            pred = unet(zt, t, cond)
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    it_s = args.steps / max(elapsed, 1e-6)
    peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
    print(f"batch_size={bs}")
    print(f"vae_fwd_bwd_s={vae_s:.3f}")
    print(f"unet_it_s={it_s:.3f}")
    print(f"peak_gb={peak:.2f}")
    print("Pick batch_size and grad_accum so peak_gb < ~20 on a 24 GB 4090, then set max_steps from it/s.")
    days_per_200k = (200_000 / max(it_s, 1e-6)) / 86400.0
    print(f"200k UNet steps at this it/s ≈ {days_per_200k:.2f} days (ignores data loading).")


if __name__ == "__main__":
    main()
