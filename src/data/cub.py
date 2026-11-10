from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import bbox_crop, pil_to_minus1_1, resize_short_side_center_crop


def _read_id_map(path: Path) -> dict[int, str]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, rest = line.split(" ", 1)
            mapping[int(idx)] = rest
    return mapping


def _read_split(path: Path) -> dict[int, int]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            idx, flag = line.strip().split()
            mapping[int(idx)] = int(flag)
    return mapping


def _read_bboxes(path: Path) -> dict[int, tuple[float, float, float, float]]:
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            idx, x, y, w, h = line.strip().split()
            mapping[int(idx)] = (float(x), float(y), float(w), float(h))
    return mapping


def find_cub_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    candidates = [
        root,
        root / "CUB_200_2011",
        root / "CUB_200_2011" / "CUB_200_2011",
    ]
    for cand in candidates:
        if (cand / "images.txt").is_file() and (cand / "images").is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find CUB images.txt under {root}. "
        "Pass --data_root pointing at CUB_200_2011 (the folder that contains images.txt)."
    )


def find_caption_root(data_root: str | Path, cub_root: Path) -> Path | None:
    root = Path(data_root)
    candidates = [
        cub_root / "text",
        cub_root / "captions",
        cub_root.parent / "text",
        cub_root.parent / "captions",
        cub_root.parent / "cvpr2016_cub" / "text",
        root / "text",
        root / "captions",
        root / "birds" / "text",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def caption_path_for_image(caption_root: Path, rel_image: str) -> Path:
    stem = Path(rel_image).with_suffix(".txt")
    direct = caption_root / stem
    if direct.is_file():
        return direct
    # Reed dump sometimes uses class folder + filename only.
    nested = caption_root / Path(rel_image).name.replace(".jpg", ".txt")
    if nested.is_file():
        return nested
    return direct


class CubDataset(Dataset):
    """Official CUB split, bbox crop, 256 center crop, Reed captions."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        image_size: int = 256,
        bbox_expand: float = 1.1,
        num_captions: int = 10,
        require_captions: bool = True,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"split must be train/test/all, got {split}")
        self.cub_root = find_cub_root(data_root)
        self.caption_root = find_caption_root(data_root, self.cub_root)
        self.image_size = image_size
        self.bbox_expand = bbox_expand
        self.num_captions = num_captions
        self.require_captions = require_captions

        images = _read_id_map(self.cub_root / "images.txt")
        split_map = _read_split(self.cub_root / "train_test_split.txt")
        bboxes = _read_bboxes(self.cub_root / "bounding_boxes.txt")

        self.samples: list[dict] = []
        self.skipped_captions = 0
        want = None if split == "all" else (1 if split == "train" else 0)
        for image_id, rel in images.items():
            if want is not None and split_map.get(image_id, -1) != want:
                continue
            image_path = self.cub_root / "images" / rel
            if not image_path.is_file():
                continue
            captions = self._load_captions(rel)
            if require_captions and not captions:
                self.skipped_captions += 1
                continue
            self.samples.append(
                {
                    "image_id": image_id,
                    "rel": rel,
                    "path": image_path,
                    "bbox": bboxes.get(image_id),
                    "captions": captions,
                }
            )
        if not self.samples:
            hint = ""
            if require_captions and self.caption_root is None:
                hint = " Caption folder not found (expected text/ next to CUB_200_2011)."
            raise RuntimeError(f"No CUB samples for split={split} under {self.cub_root}.{hint}")

    def _load_captions(self, rel_image: str) -> list[str]:
        if self.caption_root is None:
            return []
        path = caption_path_for_image(self.caption_root, rel_image)
        if not path.is_file():
            return []
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        lines = [ln for ln in lines if ln]
        return lines[: self.num_captions]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rec = self.samples[index]
        with Image.open(rec["path"]) as img:
            image = img.convert("RGB")
        if rec["bbox"] is not None:
            image = bbox_crop(image, rec["bbox"], self.bbox_expand)
        image = resize_short_side_center_crop(image, self.image_size)
        tensor = pil_to_minus1_1(image)
        captions = rec["captions"] or [""]
        cap_idx = torch_randint(len(captions))
        return {
            "image": tensor,
            "image_id": rec["image_id"],
            "caption": captions[cap_idx],
            "caption_idx": cap_idx,
            "captions": captions,
            "path": str(rec["path"]),
            "rel": rec["rel"],
        }


def torch_randint(n: int) -> int:
    import torch

    if n <= 1:
        return 0
    return int(torch.randint(0, n, (1,)).item())


def cub_collate(batch: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "image_id": torch.tensor([b["image_id"] for b in batch], dtype=torch.long),
        "caption": [b["caption"] for b in batch],
        "caption_idx": torch.tensor([b["caption_idx"] for b in batch], dtype=torch.long),
        "captions": [b["captions"] for b in batch],
        "path": [b["path"] for b in batch],
        "rel": [b["rel"] for b in batch],
    }
