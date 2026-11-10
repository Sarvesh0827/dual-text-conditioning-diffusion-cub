from __future__ import annotations

import torch
from torch import nn


class BirdPartAttentionLoss(nn.Module):
    """Optional CUB part-keypoint attention supervision. Not implemented in v1."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Bird-part attention loss is a v1 stub. Keep use_part_loss: false until "
            "experiment C already produces recognizable 256 birds."
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
