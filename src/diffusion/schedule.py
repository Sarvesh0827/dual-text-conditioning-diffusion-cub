from __future__ import annotations

import math

import torch


class CosineSchedule:
    """Improved DDPM cosine schedule (Nichol & Dhariwal)."""

    def __init__(self, timesteps: int = 1000, s: float = 0.008, device: torch.device | None = None) -> None:
        self.timesteps = timesteps
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = (f / f[0]).clamp(min=1e-6, max=1.0)
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.999).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.clamp(min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device | str) -> "CosineSchedule":
        for name, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def gather(self, name: str, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        tensor = getattr(self, name)
        out = tensor.gather(0, t.long())
        return out.view(-1, *([1] * (len(x_shape) - 1)))
