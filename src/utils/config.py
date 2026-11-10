from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def latent_scale_from_ckpt(ckpt: dict) -> float:
    if "sigma" in ckpt:
        return float(ckpt["sigma"])
    if "latent_scale" in ckpt:
        return float(ckpt["latent_scale"])
    raise KeyError(
        "VAE checkpoint has no 'sigma'. Run scripts/eval_vae.py so diffusion can use z = μ / σ."
    )
