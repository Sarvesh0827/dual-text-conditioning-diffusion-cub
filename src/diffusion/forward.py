from __future__ import annotations

import torch

from src.diffusion.schedule import CosineSchedule


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: CosineSchedule, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form forward diffusion: x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = schedule.gather("sqrt_alphas_cumprod", t, x0.shape)
    sqrt_om = schedule.gather("sqrt_one_minus_alphas_cumprod", t, x0.shape)
    return sqrt_ab * x0 + sqrt_om * noise, noise
