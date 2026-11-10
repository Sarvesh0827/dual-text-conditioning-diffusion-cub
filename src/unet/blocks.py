from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.unet.attention import SpatialCrossAttention, SpatialSelfAttention


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaGNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.in_norm = nn.GroupNorm(32, in_ch)
        self.in_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, 2 * out_ch)
        self.out_norm = nn.GroupNorm(32, out_ch)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_conv(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetLevel(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_ch: int,
        context_dim: int,
        num_res_blocks: int,
        num_heads: int,
        dropout: float,
        self_attn: bool,
        cross_attn: bool,
    ) -> None:
        super().__init__()
        blocks = []
        attn = []
        ch = in_ch
        for i in range(num_res_blocks):
            blocks.append(AdaGNResBlock(ch, out_ch, emb_ch, dropout))
            ch = out_ch
            layer_attn = nn.ModuleDict()
            if self_attn:
                layer_attn["self"] = SpatialSelfAttention(out_ch, num_heads)
            if cross_attn:
                layer_attn["cross"] = SpatialCrossAttention(out_ch, context_dim, num_heads)
            attn.append(layer_attn)
        self.blocks = nn.ModuleList(blocks)
        self.attn = nn.ModuleList(attn)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        context: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        maps: list[torch.Tensor] = []
        for block, attn in zip(self.blocks, self.attn):
            x = block(x, emb)
            if "self" in attn:
                x = attn["self"](x)
            if "cross" in attn:
                if return_attn:
                    x, a = attn["cross"](x, context, return_attn=True)
                    maps.append(a)
                else:
                    x = attn["cross"](x, context)
        return x, maps
