from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(p.data, alpha=1.0 - d)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(b.data)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow.load_state_dict(state["shadow"])
