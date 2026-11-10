#!/usr/bin/env python3
"""Download CUB-200-2011 images and Reed captions into --data_root."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
# Reed et al. CVPR 2016 CUB captions (Google Drive).
CAPTION_DRIVE_ID = "0B0ywwgffWnLLL0itM2tzT2x1N1k"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or str(archive).endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)
    else:
        with tarfile.open(archive, "r:*") as tar:
            tar.extractall(dest)


def _try_gdown(file_id: str, dest: Path) -> bool:
    try:
        import gdown
    except ImportError:
        print("gdown not installed; skip Drive caption download. pip install gdown")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(dest), quiet=False)
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        print(f"gdown failed: {exc}")
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/cub")
    args = p.parse_args()
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    cub_tar = root / "CUB_200_2011.tgz"
    if not (root / "CUB_200_2011" / "images.txt").is_file() and not (root / "images.txt").is_file():
        print(f"Downloading CUB from {CUB_URL}")
        try:
            _download(CUB_URL, cub_tar)
            _extract(cub_tar, root)
        except Exception as exc:  # noqa: BLE001
            print("Automatic CUB download failed.")
            print("Manual: get CUB_200_2011.tgz from https://www.vision.caltech.edu/datasets/cub_200_2011/")
            print(f"and extract it under {root}")
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        print("CUB images already present.")

    caption_ok = False
    for cand in [
        root / "text",
        root / "CUB_200_2011" / "text",
        root / "cvpr2016_cub" / "text",
        root / "birds" / "text",
    ]:
        if cand.is_dir() and any(cand.rglob("*.txt")):
            caption_ok = True
            print(f"Captions found at {cand}")
            break

    if not caption_ok:
        cap_archive = root / "cvpr2016_cub.tar.gz"
        print("Trying Reed caption archive via gdown...")
        if _try_gdown(CAPTION_DRIVE_ID, cap_archive):
            _extract(cap_archive, root)
            caption_ok = True

    if not caption_ok:
        print("\nCaptions were not downloaded automatically.")
        print("Get Reed CUB captions (10 lines per image) and place them so this exists:")
        print(f"  {root}/text/<class_folder>/<image_stem>.txt")
        print("or")
        print(f"  {root}/CUB_200_2011/text/<class_folder>/<image_stem>.txt")
        print("AttnGAN birds.zip / Reed cvpr2016_cub both work.")
        print("Google Drive file id (Reed):", CAPTION_DRIVE_ID)
        sys.exit(1)

    print("Done. Point --data_root at the folder that contains images.txt")
    print("Typical:", root / "CUB_200_2011")


if __name__ == "__main__":
    main()
