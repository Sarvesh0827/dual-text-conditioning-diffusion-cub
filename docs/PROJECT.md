# Dual-Conditioned Latent Diffusion for Fine-Grained Text-to-Image Synthesis on CUB-200 at 256×256

Train a text-to-image generator from scratch in PyTorch that produces 256×256 bird images from captions, and measure whether global + local text conditioning beats either pathway alone.

## 1. Domain / problem

Fine-grained conditional image generation (text-to-image). CUB birds look alike; captions must control parts and colors (“red crown”, “black throat”), not just “a bird.”

Problems:

- Fine-grained attribute binding
- 256×256 generation on one GPU (pixel-space DDPM is too heavy)
- Fair comparison of text-fusion designs on a small dataset

We are not trying to beat Stable Diffusion on the open web. Latent compression is the LDM backbone (Rombach et al.), not a claimed invention.

## 2. Baselines and gaps

| Line | Models | Limitation here |
| --- | --- | --- |
| GAN T2I | StackGAN, AttnGAN, DM-GAN, DF-GAN | Unstable; artifacts |
| Pixel diffusion | DDPM, ADM, GLIDE | Too costly at 256 + text on one 4090 |
| Latent diffusion | LDM / Stable Diffusion | Usually a black box |
| Dual text fusion | Imagen | Not ablated on CUB / limited compute |

CUB 256 literature anchors: AttnGAN FID 23.98 / IS 4.36; DF-GAN FID 14.81 / IS 5.10; VQ-Diffusion FID 10.32.

## 3. Dataset and metrics

CUB-200-2011 + Reed 10 captions/image. Official split. Train/eval at 256×256.

Metrics at 256: FID, IS, CLIPScore / R-precision, color/part swap CLIP. VAE gate: PSNR, SSIM, LPIPS.

Required success: recognizable 256 birds, prompts change parts/colors, A vs B vs C table.

## 4. Approach

From scratch: cosine forward process, ε-UNet, DDPM/DDIM, timestep AdaGN, CFG. Frozen CLIP text encoder only.

VAE f=4: 256×256×3 → 64×64×4. Dual text flags on the same UNet (identical param count). Experiment C is the main model. Bird-part loss is a v1 stub, default off.

## 5. Experiments

| ID | Global AdaGN | Local cross-attn |
| --- | --- | --- |
| A | off | on |
| B | on | off |
| C | on | on |

Same VAE, cached latents, cached CLIP, steps, sampling. Hypothesis: C ≥ A and C ≥ B on CLIPScore/swap; FID should not collapse.

Train C first, then A and B under a matched budget set after `scripts/bench_4090.py`.
