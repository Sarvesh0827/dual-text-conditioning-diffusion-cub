#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import minus1_1_to_pil
from src.diffusion.sample import ddim_sample, ddpm_sample
from src.diffusion.schedule import CosineSchedule
from src.text.clip_embed import CLIPTextEmbedder
from src.unet.model import unet_from_config
from src.utils.ckpt import load_checkpoint
from src.utils.config import latent_scale_from_ckpt, load_yaml
from src.utils.device import get_device
from src.vae.model import AutoencoderKL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/unet_C.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="samples/demo.png")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--guidance_w", type=float, default=None)
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = get_device()
    torch.manual_seed(args.seed)
    unet_ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    vae_ckpt = load_checkpoint(args.vae_ckpt, map_location="cpu")
    sigma = latent_scale_from_ckpt(vae_ckpt)

    unet = unet_from_config(unet_ckpt.get("config", cfg))
    unet.load_state_dict(unet_ckpt["model"] if "model" in unet_ckpt else unet_ckpt)
    unet.to(device).eval()

    vae_cfg = unet_ckpt.get("config", cfg)
    vae = AutoencoderKL(latent_channels=int(vae_cfg.get("latent_channels", 4)))
    vae.load_state_dict(vae_ckpt["model"] if "model" in vae_ckpt else vae_ckpt)
    vae.to(device).eval()

    clip = CLIPTextEmbedder(cfg.get("clip_model", "openai/clip-vit-base-patch32")).to(device)
    prompts = [args.prompt] * args.n
    cond = clip(prompts, device=device)
    uncond = unet.make_uncond(args.n, cond.tokens.size(1), device, cond.tokens.dtype)
    schedule = CosineSchedule(int(cfg.get("timesteps", 1000)), device=device)
    shape = (args.n, int(cfg.get("latent_channels", 4)), int(cfg.get("latent_size", 64)), int(cfg.get("latent_size", 64)))
    w = float(args.guidance_w if args.guidance_w is not None else cfg.get("guidance_w", 3.0))

    def model_fn(z, t, text):
        return unet(z, t, text)

    if args.sampler == "ddim":
        z = ddim_sample(
            model_fn,
            schedule,
            shape,
            cond,
            uncond,
            steps=int(cfg.get("ddim_steps", 50)),
            eta=float(cfg.get("ddim_eta", 0.0)),
            guidance_w=w,
            device=device,
        )
    else:
        z = ddpm_sample(model_fn, schedule, shape, cond, uncond, guidance_w=w, device=device)
    images = vae.decode(z * sigma).clamp(-1, 1)

    from PIL import Image

    pils = [minus1_1_to_pil(images[i]) for i in range(args.n)]
    canvas = Image.new("RGB", (pils[0].width * args.n, pils[0].height))
    for i, im in enumerate(pils):
        canvas.paste(im, (i * im.width, 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
