from __future__ import annotations

from PIL import Image

import torch
from torchvision.transforms import functional as TF


def bbox_crop(image: Image.Image, bbox: tuple[float, float, float, float], expand: float) -> Image.Image:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    w *= expand
    h *= expand
    x0 = max(0, int(round(cx - w / 2.0)))
    y0 = max(0, int(round(cy - h / 2.0)))
    x1 = min(image.width, int(round(cx + w / 2.0)))
    y1 = min(image.height, int(round(cy + h / 2.0)))
    if x1 <= x0 or y1 <= y0:
        return image
    return image.crop((x0, y0, x1, y1))


def resize_short_side_center_crop(image: Image.Image, size: int) -> Image.Image:
    w, h = image.size
    if min(w, h) == 0:
        raise ValueError("invalid image size")
    scale = size / float(min(w, h))
    nw = max(size, int(round(w * scale)))
    nh = max(size, int(round(h * scale)))
    image = image.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return image.crop((left, top, left + size, top + size))


def pil_to_minus1_1(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image.convert("RGB"))  # [0, 1]
    return tensor * 2.0 - 1.0


def minus1_1_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().clamp(-1, 1).cpu()
    tensor = (tensor + 1.0) / 2.0
    return TF.to_pil_image(tensor)
