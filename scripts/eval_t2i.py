#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cub import CubDataset
from src.data.transforms import minus1_1_to_pil
from src.diffusion.sample import ddim_sample
from src.diffusion.schedule import CosineSchedule
from src.text.clip_embed import CLIPTextEmbedder
from src.unet.model import unet_from_config
from src.utils.ckpt import load_checkpoint
from src.utils.config import latent_scale_from_ckpt, load_yaml
from src.utils.device import get_device
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--unet_config", default="configs/unet_C.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="runs/unet_C/eval")
    p.add_argument("--n_fid", type=int, default=None, help="Override. Use 1000 first; 30000 for paper tables.")
    p.add_argument("--skip_fid", action="store_true")
    return p.parse_args()


def load_models(unet_cfg, unet_ckpt, vae_ckpt, device):
    sigma = latent_scale_from_ckpt(vae_ckpt)
    unet = unet_from_config(unet_ckpt.get("config", unet_cfg))
    unet.load_state_dict(unet_ckpt["model"] if "model" in unet_ckpt else unet_ckpt)
    unet.to(device).eval()
    vae = AutoencoderKL(latent_channels=int(unet_cfg.get("latent_channels", 4)))
    vae.load_state_dict(vae_ckpt["model"] if "model" in vae_ckpt else vae_ckpt)
    vae.to(device).eval()
    clip = CLIPTextEmbedder(unet_cfg.get("clip_model", "openai/clip-vit-base-patch32")).to(device)
    return unet, vae, clip, sigma


@torch.no_grad()
def generate(unet, vae, clip, schedule, prompts, sigma, cfg, device, guidance_w: float) -> torch.Tensor:
    cond = clip(prompts, device=device)
    uncond = unet.make_uncond(len(prompts), cond.tokens.size(1), device, cond.tokens.dtype)
    shape = (
        len(prompts),
        int(cfg.get("latent_channels", 4)),
        int(cfg.get("latent_size", 64)),
        int(cfg.get("latent_size", 64)),
    )

    def model_fn(z, t, text):
        return unet(z, t, text)

    z = ddim_sample(
        model_fn,
        schedule,
        shape,
        cond,
        uncond,
        steps=int(cfg.get("ddim_steps", 50)),
        eta=float(cfg.get("ddim_eta", 0.0)),
        guidance_w=guidance_w,
        device=device,
        progress=False,
    )
    return vae.decode(z * sigma).clamp(-1, 1)


def clip_score_images(images: torch.Tensor, prompts: list[str], device) -> float:
    """CLIP cosine similarity using OpenAI CLIP image+text (eval only)."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    pils = [minus1_1_to_pil(images[i]) for i in range(images.size(0))]
    inputs = processor(text=prompts, images=pils, return_tensors="pt", padding=True, truncation=True).to(device)
    out = model(**inputs)
    img = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    txt = torch.nn.functional.normalize(out.text_embeds, dim=-1)
    return float((img * txt).sum(dim=-1).mean())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    eval_cfg = load_yaml(args.config)
    unet_cfg = load_yaml(args.unet_config)
    cfg = {**unet_cfg, **eval_cfg}
    device = get_device()
    out_dir = Path(args.out_dir)
    gen_dir = out_dir / "generated"
    real_dir = out_dir / "real"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    unet_ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    vae_ckpt = load_checkpoint(args.vae_ckpt, map_location="cpu")
    unet, vae, clip, sigma = load_models(cfg, unet_ckpt, vae_ckpt, device)
    schedule = CosineSchedule(int(cfg.get("timesteps", 1000)), device=device)
    w_default = float(cfg.get("default_guidance_w", 3.0))

    # Guidance qualitative grid
    prompt = "a small bird with a red crown and a black throat"
    grid_imgs = []
    for w in cfg.get("guidance_grid", [1.0, 3.0, 5.0, 7.5]):
        img = generate(unet, vae, clip, schedule, [prompt], sigma, cfg, device, float(w))[0]
        grid_imgs.append(minus1_1_to_pil(img))
    canvas = Image.new("RGB", (256 * len(grid_imgs), 256))
    for i, im in enumerate(grid_imgs):
        canvas.paste(im, (i * 256, 0))
    canvas.save(out_dir / "guidance_grid.png")

    # Color/part swap
    swap = list(cfg.get("swap_prompts", []))
    swap_scores = []
    if len(swap) >= 2:
        imgs = generate(unet, vae, clip, schedule, swap, sigma, cfg, device, w_default)
        for i, pr in enumerate(swap):
            minus1_1_to_pil(imgs[i]).save(out_dir / f"swap_{i}.png")
        score = clip_score_images(imgs, swap, device)
        swapped = swap[1::2] + swap[::2]
        if len(swapped) != len(swap):
            swapped = list(reversed(swap))
        anti = clip_score_images(imgs, swapped[: imgs.size(0)], device)
        swap_scores = {"aligned_clip": score, "swapped_clip": anti, "prompts": swap}
        print("swap", swap_scores)

    n_fid = int(args.n_fid if args.n_fid is not None else cfg.get("n_fid_samples", 1000))
    test_ds = CubDataset(args.data_root, split="test", require_captions=True)
    n_fid = min(n_fid, len(test_ds))
    bs = int(cfg.get("batch_size", 8))
    prompts_used = []
    for start in tqdm(range(0, n_fid, bs), desc="gen-fid"):
        end = min(start + bs, n_fid)
        recs = [test_ds[i] for i in range(start, end)]
        prompts = [r["caption"] for r in recs]
        prompts_used.extend(prompts)
        imgs = generate(unet, vae, clip, schedule, prompts, sigma, cfg, device, w_default)
        for j, rec in enumerate(recs):
            minus1_1_to_pil(imgs[j]).save(gen_dir / f"{rec['image_id']:05d}.png")
            minus1_1_to_pil(rec["image"]).save(real_dir / f"{rec['image_id']:05d}.png")

    metrics = {
        "n": n_fid,
        "guidance_w": w_default,
        "experiment": unet_cfg.get("experiment"),
        "n_params": unet_ckpt.get("n_params"),
        "swap": swap_scores,
    }
    if not args.skip_fid and n_fid >= 64:
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths

            fid = calculate_fid_given_paths([str(real_dir), str(gen_dir)], batch_size=32, device=str(device), dims=2048)
            metrics["fid"] = float(fid)
            print("FID", fid)
        except Exception as exc:  # noqa: BLE001
            metrics["fid_error"] = str(exc)
            print("FID failed:", exc)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Wrote guidance_grid.png and generated/ under", out_dir)


if __name__ == "__main__":
    main()
