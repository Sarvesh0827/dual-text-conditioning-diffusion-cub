from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.cuda.amp import GradScaler


def autocast_context(device: torch.device, precision: str):
    if precision == "amp_fp16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "amp_bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_scaler(device: torch.device, precision: str) -> GradScaler | None:
    if precision == "amp_fp16" and device.type == "cuda":
        return GradScaler()
    return None


def optimizer_step(
    scaler: GradScaler | None,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    max_grad_norm: float = 1.0,
) -> float:
    if scaler is None:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
