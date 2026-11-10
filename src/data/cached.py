from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedLatentClipDataset(Dataset):
    """Paired VAE latents (already /σ) and CLIP features for UNet training."""

    def __init__(self, latent_path: str | Path, clip_path: str | Path) -> None:
        latent = torch.load(latent_path, map_location="cpu", weights_only=False)
        clip = torch.load(clip_path, map_location="cpu", weights_only=False)
        self.sigma = float(latent["sigma"])
        self.image_ids = latent["image_ids"]
        self.z = latent["z"]
        clip_ids = clip["image_ids"]
        if not torch.equal(self.image_ids, clip_ids):
            index = {int(i): k for k, i in enumerate(clip_ids.tolist())}
            order = [index[int(i)] for i in self.image_ids.tolist()]
            self.tokens = clip["tokens"][order]
            self.pooled = clip["pooled"][order]
            self.input_ids = clip["input_ids"][order]
        else:
            self.tokens = clip["tokens"]
            self.pooled = clip["pooled"]
            self.input_ids = clip["input_ids"]
        if len(self.z) != len(self.tokens):
            raise ValueError("latent/clip cache size mismatch")

    def __len__(self) -> int:
        return int(self.z.size(0))

    def __getitem__(self, index: int) -> dict:
        n_caps = self.tokens.size(1)
        k = int(torch.randint(0, n_caps, (1,)).item())
        return {
            "z": self.z[index],
            "tokens": self.tokens[index, k].float(),
            "pooled": self.pooled[index, k].float(),
            "input_ids": self.input_ids[index, k],
            "image_id": int(self.image_ids[index]),
            "caption_idx": k,
        }


def cached_collate(batch: list[dict]) -> dict:
    return {
        "z": torch.stack([b["z"] for b in batch], 0),
        "tokens": torch.stack([b["tokens"] for b in batch], 0),
        "pooled": torch.stack([b["pooled"] for b in batch], 0),
        "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
        "image_id": torch.tensor([b["image_id"] for b in batch], dtype=torch.long),
        "caption_idx": torch.tensor([b["caption_idx"] for b in batch], dtype=torch.long),
    }
