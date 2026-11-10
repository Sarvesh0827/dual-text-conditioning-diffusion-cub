from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VAELoss(nn.Module):
    def __init__(self, kl_weight: float = 1e-6, lpips_weight: float = 0.1, gan_weight: float = 0.0) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_weight = lpips_weight
        self.gan_weight = gan_weight
        self._lpips = None
        if gan_weight and gan_weight > 0:
            raise NotImplementedError("PatchGAN is disabled in v1 (set gan_weight: 0.0).")

    def lpips(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.lpips_weight <= 0:
            return x.new_zeros(())
        if self._lpips is None:
            import lpips

            net = lpips.LPIPS(net="vgg")
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._lpips = net.to(x.device)
        if next(self._lpips.parameters()).device != x.device:
            self._lpips = self._lpips.to(x.device)
        return self._lpips(x, y).mean()

    def forward(self, rec: dict[str, torch.Tensor], x: torch.Tensor) -> dict[str, torch.Tensor]:
        recon = rec["recon"]
        mean = rec["mean"]
        logvar = rec["logvar"]
        l1 = F.l1_loss(recon, x)
        kl = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - 1.0 - logvar)
        perc = self.lpips(recon, x)
        total = l1 + self.lpips_weight * perc + self.kl_weight * kl
        return {"loss": total, "l1": l1, "kl": kl, "lpips": perc}
