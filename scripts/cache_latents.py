#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cub import CubDataset, cub_collate
from src.utils.ckpt import load_checkpoint
from src.utils.config import latent_scale_from_ckpt, load_yaml
from src.utils.device import get_device
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/vae.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="cache/latents")
    return p.parse_args()


@torch.no_grad()
def encode_split(model, loader, device, sigma: float) -> dict:
    zs, ids = [], []
    for batch in tqdm(loader, desc="encode"):
        x = batch["image"].to(device)
        mean, _ = model.encode(x)
        zs.append((mean.float() / sigma).cpu())
        ids.append(batch["image_id"].cpu())
    return {"z": torch.cat(zs, 0), "image_ids": torch.cat(ids, 0), "sigma": sigma}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    sigma = latent_scale_from_ckpt(ckpt)
    model = AutoencoderKL(
        latent_channels=int(cfg.get("latent_channels", 4)),
        channels=tuple(cfg.get("channels", [128, 256, 512, 512])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
    )
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    for split in ("train", "test"):
        ds = CubDataset(args.data_root, split=split, image_size=int(cfg["image_size"]), bbox_expand=float(cfg.get("bbox_expand", 1.1)), require_captions=False)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=cub_collate)
        payload = encode_split(model, loader, device, sigma)
        path = out_dir / f"{split}.pt"
        torch.save(payload, path)
        print(f"wrote {path}  z={tuple(payload['z'].shape)} σ={sigma:.6f}")
    (out_dir / "latent_scale.json").write_text(json.dumps({"sigma": sigma, "vae_ckpt": str(args.ckpt)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
