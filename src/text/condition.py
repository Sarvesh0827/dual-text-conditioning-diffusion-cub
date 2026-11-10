from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TextCondition:
    tokens: torch.Tensor  # [B, 77, 512]
    pooled: torch.Tensor  # [B, 512]
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
