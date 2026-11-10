from __future__ import annotations

import torch
from torch import nn


class QKVAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        context = x if context is None else context
        b, n, c = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, context.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        if return_attn:
            return out, attn
        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = QKVAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        out = self.attn(q)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out


class SpatialCrossAttention(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.kv_proj = nn.Linear(context_dim, channels) if context_dim != channels else nn.Identity()
        self.attn = QKVAttention(channels, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        q = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        ctx = self.kv_proj(context)
        if return_attn:
            out, attn = self.attn(q, ctx, return_attn=True)
            out = out.transpose(1, 2).reshape(b, c, h, w)
            return x + out, attn
        out = self.attn(q, ctx)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out
