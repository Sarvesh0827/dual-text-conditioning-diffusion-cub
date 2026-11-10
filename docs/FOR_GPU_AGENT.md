# Context for the GPU-machine coding agent

Read this entire file before changing anything. The Mac already built v1 of the code. Your job is **environment, data, training, debugging, and eval on an RTX 4090**. Do not redesign the model.

---

## One-sentence goal

Train a **from-scratch PyTorch** text-to-image generator that produces **256×256** CUB bird images from captions, and measure whether **global + local** text conditioning beats either pathway alone.

---

## 1. Problem statement / domain

**Course:** Deep Learning final project (not a homework DDPM).  
**Domain:** Fine-grained **text-to-image generation**.

CUB-200 birds look similar. A caption must control **parts and colors** (“red crown”, “black throat”, “white wing bars”), not just “a bird.”

**Problems we target**

1. Fine-grained attribute binding (words must affect the right region).
2. **256×256** generation on **one 24 GB RTX 4090**. Pixel-space DDPM at 256 is too heavy, so we use a latent model.
3. A **fair** comparison of text-fusion designs (papers rarely isolate this on a small dataset / one GPU).

**Professor constraints (already designed for)**

- Text-based image generator at **256×256 or more**.
- Architecture must be strong because diffusion is taught late in the course.
- Implement diffusion **from scratch in PyTorch**:
  - Forward process (cosine schedule, closed-form `x_t`)
  - Reverse process (ε-UNet, DDPM + DDIM)
  - Timestep conditioning (sinusoidal → AdaGN, always on)
  - Text conditioning / guidance (frozen CLIP + classifier-free guidance)

**What we are not claiming**

- We are not beating Stable Diffusion on the open web.
- Latent compression (VAE, f=4) is **LDM / Rombach et al.**, not our invention. Cite it as the backbone.
- Dual pooled + cross-attention fusion exists in **Imagen**. Our contribution is evaluating it on CUB at 256 with matched compute.

---

## 2. Baselines, metrics, success bar

| Line | Models | Role here |
| --- | --- | --- |
| GAN T2I | StackGAN, AttnGAN, DM-GAN, DF-GAN | CUB **256** numbers to cite |
| Pixel diffusion | DDPM, ADM, GLIDE | Too costly at 256 + text on one 4090 |
| Latent diffusion | LDM / Stable Diffusion | Backbone idea; do **not** load their weights |
| Dual text fusion | Imagen (pooled + cross-attn) | Motivation for experiments A/B/C |

CUB 256 literature anchors:

- AttnGAN: FID **23.98**, IS **4.36**
- DF-GAN: FID **14.81**, IS **5.10**
- VQ-Diffusion: FID **10.32**

**Dataset:** CUB-200-2011 official split + Reed **10 captions/image**. Train/eval at 256×256.  
Image pipeline (already coded): bbox expand **1.1** → resize shorter side 256 → center crop → `[-1, 1]`.

**Metrics at 256:** FID, IS (if feasible), CLIPScore / R-precision, color/part **swap** CLIP.  
VAE gate before UNet: PSNR, SSIM, LPIPS + visual check of eyes/beaks.

**Success**

- Required: recognizable 256 birds; prompts change color/parts; **A vs B vs C** table.
- Strong: FID better than AttnGAN (~24).
- Excellent: FID near DF-GAN (~15) **and** C beats A and B on CLIP/swap.

---

## 3. Locked architecture (do not redesign)

```text
256×256×3 image
        │
   VAE encoder (f=4)
        ▼
   z ∈ R^{4×64×64}     stored as z = μ / σ
        │
        │  t  → sinusoidal → MLP → AdaGN     ALWAYS ON
        │  pooled CLIP → add to AdaGN        if use_global_text
        │  CLIP tokens → cross-attention     if use_local_text
        │                                    else still run attn on learned null token
        ▼
   UNet εθ(z_t, t, text)
        │
   VAE decoder
        ▼
   256×256×3
```

**Allowed pretrained net:** CLIP ViT-B/32 **text** encoder only (`openai/clip-vit-base-patch32`).  
**Forbidden:** `diffusers`, pretrained SD/LDM UNet or VAE, MPS device.

**VAE:** L1 + 0.1 LPIPS + 1e-6 KL. PatchGAN **off**. After training, `scripts/eval_vae.py` writes **σ = std(μ)** into the checkpoint. Diffusion **must** use `z = μ / σ`. Never hardcode SD’s `0.18215`.

**UNet:** 4 levels on 64 latent, channels 128/256/384/512, 2 ResBlocks, self-attn at 16 and 8, cross-attn at 32/16/8, ~**81.13M params**. EMA 0.9999. Cosine T=1000. ε-prediction MSE. CFG `p_uncond=0.10`. Eval DDIM 50, guidance grid `{1, 3, 5, 7.5}`.

**A/B/C must keep identical parameter count.** Do not delete cross-attention for experiment B. Feed the learned null token instead.

| ID | Config | Global AdaGN | Local cross-attn | Meaning |
| --- | --- | --- | --- | --- |
| A | `configs/unet_A.yaml` | off | on | LDM-style baseline |
| B | `configs/unet_B.yaml` | on | off | global only |
| C | `configs/unet_C.yaml` | on | on | **main model — train first** |

Hypothesis: C ≥ A and C ≥ B on CLIPScore / swap; FID should not collapse.

**Bird-part attention loss:** stub only. `use_part_loss: false`. Do **not** implement unless experiment C already makes recognizable birds **and** the owners ask.

`python scripts/check_shapes.py` (already passed on Mac): VAE 65.55M, UNet A=B=C 81.13M, latent shape `(B, 4, 64, 64)`.

---

## 4. What is already implemented (this zip)

Do not recreate these files from scratch.

```text
README.md
requirements.txt
docs/PROJECT.md
docs/FOR_GPU_AGENT.md          ← this file
configs/*.yaml
src/vae/                       AutoencoderKL + loss
src/unet/                      dual-cond UNet, AdaGN, attn
src/diffusion/                 cosine, q_sample, DDPM, DDIM, CFG
src/text/                      CLIP embedder + TextCondition
src/data/                      CUB loader, caches
scripts/download_cub.py
scripts/check_shapes.py
scripts/bench_4090.py
scripts/train_vae.py
scripts/eval_vae.py            writes sigma
scripts/cache_latents.py
scripts/cache_clip.py
scripts/train_unet.py
scripts/sample.py
scripts/eval_t2i.py
```

**Not in the zip (on purpose):** `.venv`, CUB images, captions, `runs/`, `cache/`. Those must be created on the 4090.

---

## 5. Your job on the RTX 4090

### 5.1 Environment

Python 3.10–3.12. Install **CUDA PyTorch first**, then requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True` and a 4090-class GPU. If False, you are on CPU — stop.

Device helper uses `cuda` else `cpu`. Never add `mps`.

Need ~40 GB disk. First run also downloads CLIP / LPIPS VGG / Inception.

### 5.2 Data

```bash
python scripts/download_cub.py --data_root data/cub
```

Need `images.txt` + `images/` + Reed captions (`text/` with 10 lines per image).  
**Captions often fail (Google Drive).** If so, fetch Reed/AttnGAN bird captions manually and place them where `src/data/cub.py` looks (`text/` next to or inside `CUB_200_2011`). VAE can train without captions; UNet cannot.

Typical `--data_root`: `data/cub/CUB_200_2011`

### 5.3 Run order (do not skip)

```bash
python scripts/check_shapes.py

python scripts/bench_4090.py --config configs/unet_C.yaml
# then edit batch_size, grad_accum, max_steps in yaml from peak_gb and it/s
# keep peak memory under ~20 GB on a 24 GB card
# A/B/C must keep the SAME max_steps / batch / accum after you choose them

python scripts/train_vae.py --config configs/vae.yaml --data_root data/cub/CUB_200_2011 --out_dir runs/vae
python scripts/eval_vae.py  --config configs/vae.yaml --data_root data/cub/CUB_200_2011 --ckpt runs/vae/vae_ema.pt --out_dir runs/vae
# eval_vae writes sigma into the ckpt. UNet training will refuse without it.

python scripts/cache_latents.py --config configs/vae.yaml --data_root data/cub/CUB_200_2011 --ckpt runs/vae/vae_ema.pt --out_dir cache/latents
python scripts/cache_clip.py    --config configs/default.yaml --data_root data/cub/CUB_200_2011 --out_dir cache/clip

python scripts/train_unet.py --config configs/unet_C.yaml --data_root data/cub/CUB_200_2011 --vae_ckpt runs/vae/vae_ema.pt --latent_cache cache/latents --clip_cache cache/clip --out_dir runs/unet_C
# then A and B with the same caches, same step budget

python scripts/sample.py --config configs/unet_C.yaml --ckpt runs/unet_C/unet_ema.pt --vae_ckpt runs/vae/vae_ema.pt --prompt "a small bird with a red crown and a black throat" --out samples/demo.png
python scripts/eval_t2i.py --config configs/eval.yaml --unet_config configs/unet_C.yaml --ckpt runs/unet_C/unet_ema.pt --vae_ckpt runs/vae/vae_ema.pt --data_root data/cub/CUB_200_2011 --out_dir runs/unet_C/eval
```

If you retrain the VAE, **rebuild latent cache**. CLIP cache can stay if captions unchanged.

`max_steps` in yaml is an **estimate**. Set it from `bench_4090.py`. Prefer finishing **C well** over three half-trained models.

### 5.4 Allowed bugfixes

You **may**:

- Fix CUDA OOM (lower `batch_size`, raise `grad_accum`, turn on `grad_checkpoint`)
- Fix Windows paths / DataLoader `num_workers` / pin_memory
- Fix caption folder discovery if Reed dump uses a slightly different layout
- Resume from checkpoints if you add it carefully
- Logging, tqdm, TensorBoard
- Make download_cub more reliable
- FID sample count: start with `--n_fid 1000`, use 30k only for final tables

You **may not** without the owners asking:

- Switch to Diffusers / pretrained SD
- Change latent size away from 64×64×4 / f=4
- Drop 256 output
- Delete cross-attn in B to “save compute”
- Implement part-loss as the main path
- Train on CIFAR/COCO instead of CUB
- Claim VAE/LDM as a new invention in writeups

---

## 6. Presentation / report (later)

Demo 256 prompt → image. Architecture diagram with forward, reverse, timestep AdaGN, dual text. Table A vs B vs C. Guidance 1 vs 7.5. Failures (mixed colors, missing parts). Cite LDM for the latent backbone.

---

## 7. Short answers for the agent

- **What is this?** CUB 256 text-to-image latent diffusion, from scratch, dual-conditioning ablation.
- **What is done?** Full v1 code on Mac. Shapes verified. Not trained.
- **What do I do?** CUDA env, CUB+captions, bench, VAE, σ, caches, train C then A/B, sample, FID/CLIP.
- **What must stay?** f=4 VAE, 81M UNet, identical A/B/C params, CLIP frozen, no diffusers, 256 output.
