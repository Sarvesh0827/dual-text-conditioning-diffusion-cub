from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from src.diffusion.schedule import CosineSchedule
from src.text.condition import TextCondition


def _cfg_eps(
    model: Callable,
    z: torch.Tensor,
    t: torch.Tensor,
    cond: TextCondition,
    uncond: TextCondition,
    guidance_w: float,
) -> torch.Tensor:
    if guidance_w <= 1.0:
        return model(z, t, cond)
    z_in = torch.cat([z, z], dim=0)
    t_in = torch.cat([t, t], dim=0)
    tokens = torch.cat([uncond.tokens, cond.tokens], dim=0)
    pooled = torch.cat([uncond.pooled, cond.pooled], dim=0)
    mixed = TextCondition(tokens=tokens, pooled=pooled)
    eps_uncond, eps_cond = model(z_in, t_in, mixed).chunk(2, dim=0)
    return eps_uncond + guidance_w * (eps_cond - eps_uncond)


@torch.no_grad()
def ddpm_sample(
    model: Callable,
    schedule: CosineSchedule,
    shape: tuple[int, ...],
    cond: TextCondition,
    uncond: TextCondition,
    guidance_w: float = 3.0,
    device: torch.device | None = None,
    progress: bool = True,
) -> torch.Tensor:
    device = device or cond.tokens.device
    z = torch.randn(shape, device=device)
    steps = range(schedule.timesteps - 1, -1, -1)
    if progress:
        steps = tqdm(steps, desc="DDPM", leave=False)
    for i in steps:
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = _cfg_eps(model, z, t, cond, uncond, guidance_w)
        beta = schedule.gather("betas", t, z.shape)
        sqrt_one_minus = schedule.gather("sqrt_one_minus_alphas_cumprod", t, z.shape)
        sqrt_recip_alpha = schedule.gather("sqrt_recip_alphas", t, z.shape)
        mean = sqrt_recip_alpha * (z - beta / sqrt_one_minus * eps)
        if i == 0:
            z = mean
        else:
            noise = torch.randn_like(z)
            logvar = schedule.gather("posterior_log_variance_clipped", t, z.shape)
            z = mean + torch.exp(0.5 * logvar) * noise
    return z


@torch.no_grad()
def ddim_sample(
    model: Callable,
    schedule: CosineSchedule,
    shape: tuple[int, ...],
    cond: TextCondition,
    uncond: TextCondition,
    steps: int = 50,
    eta: float = 0.0,
    guidance_w: float = 3.0,
    device: torch.device | None = None,
    progress: bool = True,
) -> torch.Tensor:
    device = device or cond.tokens.device
    z = torch.randn(shape, device=device)
    ts = torch.linspace(schedule.timesteps - 1, 0, steps, device=device).long()
    iterator = range(len(ts))
    if progress:
        iterator = tqdm(iterator, desc="DDIM", leave=False)
    for idx in iterator:
        i = int(ts[idx].item())
        i_prev = int(ts[idx + 1].item()) if idx + 1 < len(ts) else -1
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = _cfg_eps(model, z, t, cond, uncond, guidance_w)
        a_t = schedule.alphas_cumprod[i]
        a_prev = schedule.alphas_cumprod[i_prev] if i_prev >= 0 else z.new_ones(())
        x0 = (z - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
        if i_prev < 0:
            z = x0
            continue
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps
        noise = torch.randn_like(z) if eta > 0 else 0.0
        z = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise
    return z
