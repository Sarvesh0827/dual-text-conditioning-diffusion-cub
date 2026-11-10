#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cub import CubDataset
from src.text.clip_embed import CLIPTextEmbedder
from src.utils.config import load_yaml
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="cache/clip")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def cache_split(embedder: CLIPTextEmbedder, ds: CubDataset, device, out_path: Path, batch_size: int) -> None:
    tokens, pooled, input_ids = [], [], []
    n_caps = ds.num_captions
    image_ids = []
    cap_buf: list[str] = []
    id_buf: list[int] = []
    for rec in tqdm(ds.samples, desc=out_path.stem):
        caps = rec["captions"]
        if len(caps) < n_caps:
            caps = (caps + [""] * n_caps)[:n_caps]
        else:
            caps = caps[:n_caps]
        cap_buf.extend(caps)
        id_buf.append(rec["image_id"])
        if len(id_buf) >= batch_size:
            cond = embedder(cap_buf, device=device)
            b = len(id_buf)
            tokens.append(cond.tokens.float().cpu().view(b, n_caps, -1, cond.tokens.size(-1)))
            pooled.append(cond.pooled.float().cpu().view(b, n_caps, -1))
            input_ids.append(cond.input_ids.cpu().view(b, n_caps, -1))
            image_ids.extend(id_buf)
            cap_buf, id_buf = [], []
    if id_buf:
        cond = embedder(cap_buf, device=device)
        b = len(id_buf)
        tokens.append(cond.tokens.float().cpu().view(b, n_caps, -1, cond.tokens.size(-1)))
        pooled.append(cond.pooled.float().cpu().view(b, n_caps, -1))
        input_ids.append(cond.input_ids.cpu().view(b, n_caps, -1))
        image_ids.extend(id_buf)

    payload = {
        "image_ids": torch.tensor(image_ids, dtype=torch.long),
        "tokens": torch.cat(tokens, 0).half(),
        "pooled": torch.cat(pooled, 0).half(),
        "input_ids": torch.cat(input_ids, 0),
        "model": embedder.model.config._name_or_path if hasattr(embedder.model.config, "_name_or_path") else "clip",
    }
    torch.save(payload, out_path)
    print(f"wrote {out_path} tokens={tuple(payload['tokens'].shape)}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = CLIPTextEmbedder(cfg.get("clip_model", "openai/clip-vit-base-patch32"), max_length=int(cfg.get("text_max_length", 77))).to(device)
    for split in ("train", "test"):
        ds = CubDataset(
            args.data_root,
            split=split,
            image_size=int(cfg.get("image_size", 256)),
            bbox_expand=float(cfg.get("bbox_expand", 1.1)),
            num_captions=int(cfg.get("num_captions", 10)),
            require_captions=True,
        )
        print(f"{split}: {len(ds)} images with captions (skipped {ds.skipped_captions})")
        cache_split(embedder, ds, device, out_dir / f"{split}.pt", args.batch_size)


if __name__ == "__main__":
    main()
