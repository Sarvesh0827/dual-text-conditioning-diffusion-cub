from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.n1 = gn(in_ch)
        self.n2 = gn(out_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


def _res_stack(in_ch: int, out_ch: int, n: int) -> nn.Sequential:
    blocks = [ResBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(n)]
    return nn.Sequential(*blocks)


class AutoencoderKL(nn.Module):
    """f=4 KL-VAE: 256x256x3 -> 64x64x4. No attention in v1.

    Encoder: 256 --(down)--> 128 --(down)--> 64, then extra 512-wide processing at 64.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        channels: tuple[int, ...] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("v1 VAE expects 4 channel stages: (128, 256, 512, 512)")
        c0, c1, c2, c3 = channels
        self.latent_channels = latent_channels
        n = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.enc_0 = _res_stack(c0, c0, n)                 # 256
        self.down_0 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)  # 128
        self.enc_1 = _res_stack(c1, c1, n)                 # 128
        self.down_1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)  # 64
        self.enc_2 = _res_stack(c2, c2, n)                 # 64
        self.enc_3 = _res_stack(c2, c3, n)                 # 64
        self.enc_mid = nn.Sequential(ResBlock(c3, c3), ResBlock(c3, c3))
        self.enc_out = nn.Sequential(gn(c3), nn.SiLU(), nn.Conv2d(c3, 2 * latent_channels, 1))

        self.dec_in = nn.Conv2d(latent_channels, c3, 3, padding=1)
        self.dec_mid = nn.Sequential(ResBlock(c3, c3), ResBlock(c3, c3))
        self.dec_3 = _res_stack(c3, c3, n)                 # 64
        self.dec_2 = _res_stack(c3, c2, n)                 # 64
        self.up_1 = nn.Sequential(                          # 128
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c1, 3, padding=1),
        )
        self.dec_1 = _res_stack(c1, c1, n)
        self.up_0 = nn.Sequential(                          # 256
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, c0, 3, padding=1),
        )
        self.dec_0 = _res_stack(c0, c0, n)
        self.conv_out = nn.Sequential(gn(c0), nn.SiLU(), nn.Conv2d(c0, in_channels, 3, padding=1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_3(self.enc_2(self.down_1(self.enc_1(self.down_0(self.enc_0(self.conv_in(x)))))))
        h = self.enc_mid(h)
        mean, logvar = self.enc_out(h).chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_0(self.up_0(self.dec_1(self.up_1(self.dec_2(self.dec_3(self.dec_mid(self.dec_in(z))))))))
        return torch.tanh(self.conv_out(h))

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(std)
        return mean

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return {"recon": recon, "mean": mean, "logvar": logvar, "z": z}
