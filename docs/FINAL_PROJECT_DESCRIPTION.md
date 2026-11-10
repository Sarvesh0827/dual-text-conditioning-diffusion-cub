# Final Project Description

**Title:** Dual-Conditioned Latent Diffusion for Fine-Grained Text-to-Image Synthesis on CUB-200 at 256×256

**One-sentence goal:** Train a text-to-image generator from scratch in PyTorch that produces 256×256 bird images from captions, and measure whether global + local text conditioning beats either pathway alone.

**Course context:** Deep Learning final course project. Professor requirements already designed for: text-based image generator at **256×256 or more**, and a strong architecture because diffusion is taught late in the course.

---

## 1. Domain / Problem

**Domain:** Fine-grained conditional image generation (text-to-image).

CUB-200 birds look alike. A caption must control **parts and colors** (e.g. “red crown”, “black throat”, “white wing bars”), not just “a bird.”

**Problems we target**

1. **Fine-grained attribute binding** — words must land on the right region of the image.
2. **256×256 generation on one GPU** — pixel-space DDPM is too heavy for a single RTX 4090 with text conditioning.
3. **Fair comparison of text-fusion designs** — papers rarely isolate global vs local conditioning on a small dataset with matched compute.

We are **not** trying to beat Stable Diffusion on the open web. We are building a transparent 256-scale text-to-image system and testing one architectural hypothesis.

---

## 2. Current Baselines / Gaps

| Line | Models | Status | Limitation for this project |
| --- | --- | --- | --- |
| GAN T2I | StackGAN, AttnGAN, DM-GAN, DF-GAN | Standard CUB **256** numbers | Unstable training; artifacts; weak composition |
| Pixel diffusion | DDPM, ADM, GLIDE | Strong sample quality | Too costly at 256 + text on one 4090 |
| Latent diffusion | LDM / Stable Diffusion | Practical 256 method | Usually used as a black box; compression is already known |
| Dual text fusion | Imagen (pooled + cross-attn) | Used at huge scale | Not evaluated as a controlled ablation on CUB / limited compute |

**Gaps we fill**

1. From-scratch latent text-to-image at 256, fully implemented (forward, reverse, timestep, text guidance).
2. Matched A/B/C test of global vs local vs both.
3. Optional later: CUB-part attention supervision if the baseline already works.

We will **not** call the VAE or latent diffusion “our invention.” We cite Rombach et al. (LDM) as the backbone. Dual conditioning exists in Imagen; our contribution is evaluating it on fine-grained CUB with one GPU.

---

## 3. Dataset and Evaluation Metrics

**Dataset:** CUB-200-2011 + Reed 10 captions per image  
- 11,788 images, 200 species  
- Official train/test split  
- Train and evaluate at **256×256**

**Image pipeline:** CUB bbox expanded by 1.1 → resize shorter side to 256 → center crop 256 → normalize to `[-1, 1]`

**Optional later (not required for claim):** Oxford-102 Flowers for qualitative extras; CUB 15 part keypoints / 312 attributes only if we add the optional part-loss.

### Metrics (all at 256)

| Metric | Purpose | CUB literature anchors |
| --- | --- | --- |
| FID ↓ | Realism | AttnGAN 23.98; DF-GAN 14.81; VQ-Diffusion 10.32 |
| IS ↑ | Sample quality | AttnGAN 4.36; DF-GAN 5.10 |
| CLIPScore / R-precision ↑ | Text–image match | GALIP CLIPSim ~0.32 |
| Color/part swap CLIP | Binding | e.g. “red crown, black throat” vs swapped |
| VAE PSNR / SSIM / LPIPS | Reconstruction gate | Must pass before UNet training |

### Success bar

- **Required:** recognizable 256 birds; prompts change color/parts; A vs B vs C table
- **Strong:** FID better than AttnGAN (~24)
- **Excellent:** FID near DF-GAN (~15) **and** C beats A and B on CLIP / swap

---

## 4. Approach / Technical Novelty

### Implemented from scratch in PyTorch

No HuggingFace Diffusers. No pretrained Stable Diffusion / LDM UNet or VAE weights.

1. **Forward process** — cosine noise schedule, closed-form `x_t`
2. **Reverse process** — ε-prediction UNet, DDPM + DDIM sampling
3. **Timestep conditioning** — sinusoidal embedding → AdaGN in every ResBlock (**always on**)
4. **Text conditioning / guidance** — frozen CLIP ViT-B/32 + classifier-free guidance (`p_uncond = 0.10`)

**Allowed pretrained module:** CLIP text encoder only.  
**Trained by us:** VAE, UNet, null token, sampler, CFG, losses.

### Architecture (locked)

```text
256×256×3 image
        │
   VAE encoder (f=4)
        ▼
   z ∈ R^{4×64×64}     [stored scaled as z = μ / σ]
        │
        │  t → sinusoidal → MLP → AdaGN          ALWAYS ON
        │  pooled CLIP → add into AdaGN          if use_global_text
        │  CLIP tokens → cross-attention         if use_local_text
        │                                        else still run attn on learned null token
        ▼
   UNet εθ(z_t, t, text)
        │
   VAE decoder
        ▼
   256×256×3 image
```

**VAE:** f=4, `256 → 64×64×4`, L1 + LPIPS + KL (+ PatchGAN off by default). Latent scale `σ` is measured once after VAE training and frozen for every UNet run. Never use Stable Diffusion’s hardcoded `0.18215`.

**UNet:** 4 levels on 64×64 latent, channels 128 / 256 / 384 / 512, self-attn at 16 & 8, cross-attn from 32 down, ~**81M params**, EMA weights.

### Central contribution: dual-conditioning ablation

Same UNet backbone for all runs. Two independent flags. Timestep AdaGN never turned off. Parameter counts for A/B/C must match.

| ID | Global (pooled CLIP → AdaGN) | Local (tokens → cross-attn) | What it is |
| --- | --- | --- | --- |
| A | Off | On | Standard LDM-style text |
| B | On | Off | Global-only (Imagen pooled path) |
| C | On | On | **Main model** |

**Hypothesis:** C ≥ A and C ≥ B on CLIPScore and color/part swap; FID should not collapse.

Train under matched data, cached latents, cached CLIP features, steps, batch, and sampling settings. Train **C first**, then A and B.

### Optional extension (only after C works)

CUB keypoints → soft spatial targets on attention maps for words like “crown”, “throat”, “beak”:

`L_total = L_diffusion + λ_part * L_part`

Default: **off**. Adapts TokenCompose-style token supervision to CUB parts; we do not claim we invented attention supervision.

---

## 5. Training / Workflow Split

| Machine | Role |
| --- | --- |
| Mac | Code, configs, docs, report, import/shape checks |
| RTX 4090 (24 GB) | CUB download, VAE train, caches, UNet A/B/C, FID / CLIP / samples |

Always on for GPU runs: AMP, gradient accumulation, EMA, checkpoints.  
Before multi-day training: run `scripts/bench_4090.py` and set `batch_size`, `grad_accum`, `max_steps` from memory and it/s. The day counts in early notes are **estimates**, not commitments.

### Implementation order

1. CUB 256 pipeline + CLIP cache  
2. f=4 VAE → pass PSNR/SSIM/LPIPS + visual check; freeze `σ`  
3. Sanity latent diffusion (noise → bird-shaped samples)  
4. Modular UNet flags + CFG  
5. Train **C**  
6. Train **A** and **B** under the matched budget  
7. Eval at 256: FID, IS, CLIPScore, swap test, guidance grid  
8. Part-loss only if C already looks good  

---

## 6. Current status (as of Mac handoff)

**Done on Mac (code v1):**

- Full repo: VAE, UNet, cosine DDPM/DDIM, dual text flags, CFG, EMA, CUB loader, train/eval/cache/bench scripts
- Configs for A / B / C
- Shape check passed: VAE **65.55M** params, UNet A=B=C **81.13M** params, latent `(B, 4, 64, 64)`
- Docs: this description, GPU agent brief, README

**Not done yet (4090 work):**

- Download CUB + Reed captions
- CUDA environment
- Train VAE / UNet
- FID / CLIP final numbers
- Presentation samples

---

## 7. What we will show in the presentation

1. Live/recorded **256** prompt → image  
2. Why 256 forced latent f=4 (cite LDM; do not call it new)  
3. Architecture diagram with the four required pieces: forward, reverse, timestep AdaGN, text (global + local)  
4. Table **A vs B vs C**  
5. Same prompt at guidance **1 vs 7.5**  
6. Failures: mixed colors, missing parts  
7. Part-loss only if it exists and helps  

---

## 8. Out of scope

- Pixel-space 256 DDPM as the main model  
- HuggingFace Diffusers / pretrained SD UNet or VAE  
- COCO / LAION as the main dataset  
- Promising 512 as required (optional SR only if 256 C is done)  
- Calling latent compression or dual conditioning “invented here”  
- Making bird-part loss mandatory  

---

## 9. Files to give teammates

| File | Use |
| --- | --- |
| `docs/FINAL_PROJECT_DESCRIPTION.md` | This document — project plan for humans |
| `docs/FOR_GPU_AGENT.md` | Exact instructions for the 4090 Cursor agent |
| `docs/PROJECT.md` | Short research summary |
| `README.md` | Setup and training commands |
| `dl_final_project.zip` | Full code (no `.venv`, no CUB data, no checkpoints) |

**Tell the GPU teammate:** unzip → open `docs/FOR_GPU_AGENT.md` in Cursor → do not redesign the model → CUDA setup → CUB → bench → VAE → eval_vae (writes σ) → caches → train C then A/B → sample/eval.
