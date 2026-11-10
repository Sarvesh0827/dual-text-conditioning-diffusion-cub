from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.text.condition import TextCondition
from src.unet.attention import SpatialCrossAttention, SpatialSelfAttention
from src.unet.blocks import AdaGNResBlock, Downsample, UNetLevel, Upsample, timestep_embedding
from src.utils.config import count_params


class LatentUNet(nn.Module):
    """ε-prediction UNet on 64x64x4 latents with dual text conditioning.

    Cross-attention modules always exist so experiments A/B/C share parameter count.
    When local text is off, they run on a learned null token.
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: tuple[int, ...] = (128, 256, 384, 512),
        num_res_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        context_dim: int = 512,
        time_embed_dim: int = 128,
        self_attn_res: tuple[int, ...] = (16, 8),
        cross_attn_res: tuple[int, ...] = (32, 16, 8),
        use_global_text: bool = True,
        use_local_text: bool = True,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if len(model_channels) != 4:
            raise ValueError("v1 UNet expects 4 levels: 64, 32, 16, 8")
        self.use_global_text = use_global_text
        self.use_local_text = use_local_text
        self.grad_checkpoint = grad_checkpoint
        self.time_embed_dim = time_embed_dim
        chs = list(model_channels)
        emb_ch = chs[-1]
        self.emb_ch = emb_ch

        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )
        self.pooled_proj = nn.Linear(context_dim, emb_ch)
        self.null_pooled = nn.Parameter(torch.zeros(1, context_dim))
        self.null_token = nn.Parameter(torch.zeros(1, 1, context_dim))
        nn.init.normal_(self.null_pooled, std=0.02)
        nn.init.normal_(self.null_token, std=0.02)

        self_attn_res = set(self_attn_res)
        cross_attn_res = set(cross_attn_res)
        resolutions = [64, 32, 16, 8]

        self.enc_levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = chs[0]
        for i, (ch, res) in enumerate(zip(chs, resolutions)):
            self.enc_levels.append(
                UNetLevel(
                    in_ch,
                    ch,
                    emb_ch,
                    context_dim,
                    num_res_blocks,
                    num_heads,
                    dropout,
                    self_attn=res in self_attn_res,
                    cross_attn=res in cross_attn_res,
                )
            )
            in_ch = ch
            if i < len(chs) - 1:
                self.downs.append(Downsample(ch))

        self.mid_res1 = AdaGNResBlock(chs[-1], chs[-1], emb_ch, dropout)
        self.mid_self = SpatialSelfAttention(chs[-1], num_heads)
        self.mid_cross = SpatialCrossAttention(chs[-1], context_dim, num_heads)
        self.mid_res2 = AdaGNResBlock(chs[-1], chs[-1], emb_ch, dropout)

        self._rebuild_decoder_in_channels(
            chs,
            resolutions,
            emb_ch,
            context_dim,
            num_res_blocks,
            num_heads,
            dropout,
            self_attn_res,
            cross_attn_res,
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_channels, 3, padding=1),
        )

    def _rebuild_decoder_in_channels(
        self,
        chs: list[int],
        resolutions: list[int],
        emb_ch: int,
        context_dim: int,
        num_res_blocks: int,
        num_heads: int,
        dropout: float,
        self_attn_res: set[int],
        cross_attn_res: set[int],
    ) -> None:
        """Decoder in_ch = current channels + skip channels. Current starts at 512 (bottleneck)."""
        self.dec_levels = nn.ModuleList()
        self.ups = nn.ModuleList()
        cur_ch = chs[-1]
        for i in range(len(chs) - 1, -1, -1):
            skip_ch = chs[i]
            res = resolutions[i]
            self.dec_levels.append(
                UNetLevel(
                    cur_ch + skip_ch,
                    chs[i],
                    emb_ch,
                    context_dim,
                    num_res_blocks,
                    num_heads,
                    dropout,
                    self_attn=res in self_attn_res,
                    cross_attn=res in cross_attn_res,
                )
            )
            cur_ch = chs[i]
            if i > 0:
                self.ups.append(Upsample(cur_ch))

    def _emb_and_context(
        self,
        t: torch.Tensor,
        cond: TextCondition,
        drop_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = t.size(0)
        n_tokens = cond.tokens.size(1)
        pooled = cond.pooled
        tokens = cond.tokens
        null_p = self.null_pooled.expand(b, -1)
        null_t = self.null_token.expand(b, n_tokens, -1)
        if drop_cond is not None:
            mask = drop_cond.view(b, 1).to(dtype=torch.bool)
            pooled = torch.where(mask, null_p, pooled)
            tokens = torch.where(mask.unsqueeze(-1), null_t, tokens)
        if not self.use_local_text:
            tokens = null_t
        t_emb = self.time_mlp(timestep_embedding(t, self.time_embed_dim))
        if self.use_global_text:
            t_emb = t_emb + self.pooled_proj(pooled)
        return t_emb, tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: TextCondition,
        drop_cond: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        emb, context = self._emb_and_context(t, cond, drop_cond)
        attn_maps: list[torch.Tensor] = []
        h = self.conv_in(x)
        skips: list[torch.Tensor] = []
        for i, level in enumerate(self.enc_levels):
            h, maps = self._run_level(level, h, emb, context, return_attn)
            attn_maps.extend(maps)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.mid_res1(h, emb)
        h = self.mid_self(h)
        if return_attn:
            h, mid_a = self.mid_cross(h, context, return_attn=True)
            attn_maps.append(mid_a)
        else:
            h = self.mid_cross(h, context)
        h = self.mid_res2(h, emb)

        for i, level in enumerate(self.dec_levels):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h, maps = self._run_level(level, h, emb, context, return_attn)
            attn_maps.extend(maps)
            if i < len(self.ups):
                h = self.ups[i](h)

        out = self.conv_out(h)
        if return_attn:
            return out, attn_maps
        return out

    def _run_level(self, level, h, emb, context, return_attn):
        if self.grad_checkpoint and self.training:
            def fn(h_, emb_, ctx_):
                y, _ = level(h_, emb_, ctx_, return_attn=False)
                return y

            y = checkpoint(fn, h, emb, context, use_reentrant=False)
            return y, []
        return level(h, emb, context, return_attn=return_attn)

    def make_uncond(self, batch: int, n_tokens: int, device: torch.device, dtype: torch.dtype) -> TextCondition:
        return TextCondition(
            tokens=self.null_token.expand(batch, n_tokens, -1).to(device=device, dtype=dtype),
            pooled=self.null_pooled.expand(batch, -1).to(device=device, dtype=dtype),
        )


def count_unet_params(model: nn.Module) -> int:
    return count_params(model)


def unet_from_config(cfg: dict) -> LatentUNet:
    return LatentUNet(
        in_channels=cfg.get("latent_channels", 4),
        model_channels=tuple(cfg.get("model_channels", [128, 256, 384, 512])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        context_dim=512,
        time_embed_dim=int(cfg.get("time_embed_dim", 128)),
        self_attn_res=tuple(cfg.get("self_attn_res", [16, 8])),
        cross_attn_res=tuple(cfg.get("cross_attn_res", [32, 16, 8])),
        use_global_text=bool(cfg.get("use_global_text", True)),
        use_local_text=bool(cfg.get("use_local_text", True)),
        grad_checkpoint=bool(cfg.get("grad_checkpoint", False)),
    )
