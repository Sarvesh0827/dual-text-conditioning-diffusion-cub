#!/usr/bin/env python3
"""Mac-safe smoke test: build VAE/UNet, check shapes, print param counts. No CUB required."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diffusion.forward import q_sample
from src.diffusion.schedule import CosineSchedule
from src.text.condition import TextCondition
from src.unet.model import count_unet_params, unet_from_config
from src.utils.config import load_yaml
from src.vae.model import AutoencoderKL


def main() -> None:
    device = torch.device("cpu")
    vae = AutoencoderKL()
    x = torch.randn(2, 3, 256, 256)
    out = vae(x)
    assert out["mean"].shape == (2, 4, 64, 64), out["mean"].shape
    assert out["recon"].shape == (2, 3, 256, 256), out["recon"].shape
    print(f"VAE OK  params={count_unet_params(vae)/1e6:.2f}M  latent={tuple(out['mean'].shape)}")

    counts = {}
    for name in ("A", "B", "C"):
        cfg = load_yaml(ROOT / "configs" / f"unet_{name}.yaml")
        unet = unet_from_config(cfg)
        counts[name] = count_unet_params(unet)
        z = torch.randn(2, 4, 64, 64)
        t = torch.randint(0, 1000, (2,))
        cond = TextCondition(tokens=torch.randn(2, 77, 512), pooled=torch.randn(2, 512))
        drop = torch.tensor([True, False])
        eps = unet(z, t, cond, drop_cond=drop)
        assert eps.shape == z.shape, eps.shape
        print(
            f"UNet {name} OK  params={counts[name]/1e6:.2f}M  "
            f"global={cfg['use_global_text']} local={cfg['use_local_text']}"
        )
    if len(set(counts.values())) != 1:
        raise SystemExit(f"A/B/C param counts differ: {counts}")
    print("A/B/C parameter counts match.")

    schedule = CosineSchedule(1000)
    zt, noise = q_sample(z, t, schedule)
    assert zt.shape == z.shape
    print("cosine forward OK")
    print("check_shapes passed")


if __name__ == "__main__":
    main()
