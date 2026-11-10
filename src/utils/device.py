from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Prefer CUDA. CPU fallback. Never MPS — training target is an RTX 4090."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_name(device: torch.device | None = None) -> str:
    device = device or get_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"
