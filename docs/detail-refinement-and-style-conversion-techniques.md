# Detail refinement & style conversion techniques for ComfyUI (FLUX.2 / Klein 9B)

This document distills the techniques used across our reference workflows in
[`user/default/workflows/references/`](../user/default/workflows/references/),
cross-checked against current (2026) community and primary-source research, for
two jobs:

1. **Detail refinement & high resolution** — take a low-res or soft input image
   and plausibly add fine detail (skin/pores, cloth/garment texture, hair, facial
   expression) while pushing to 2K–4K.
2. **Style conversion / repaint** — convert an image to a different style
   (anime/2D ↔ photoreal, 3D ↔ 2D, 2.5D → 3D) while preserving composition,
   pose, and identity.

The goal is not to run the reference workflows verbatim, but to extract the
transferable patterns and parameters so we can rebuild them cleanly.

## 0. The shared stack & two facts that govern everything

Every reference workflow targets the **FLUX.2 family**, almost all on **FLUX.2
Klein 9B** (the distilled instruction/edit model — the FLUX.2-era successor to
FLUX.1 Kontext). The recurring stack:

- **Diffusion:** `flux-2-klein-9b-fp8.safetensors` / `flux-2-klein-base-9b-fp8.safetensors`
  / `flux-2-klein-9b-Q8_0.gguf` (plus finetunes: `moody-desire`, `gonzalomoKlein`,
  `pornmasterFlux2Klein`; a couple use `flux2_dev_fp8mixed`).
- **Text encoder:** `qwen_3_8b_fp8mixed.safetensors` via `CLIPLoader` type **`flux2`**
  (4B path uses `qwen_3_4b`). FLUX.2 *dev* instead uses `mistral_3_small_flux2_fp8`.
- **VAE:** `flux2-vae.safetensors`.

**Fact 1 — Klein/Kontext editing is NOT img2img.** The reference image is
VAE-encoded and injected into the *conditioning* through `ReferenceLatent`; the
sampler runs at **denoise 1.0** on an `EmptyFlux2LatentImage`. Composition and
identity are held by the reference latent, not by a low denoise. The familiar
"denoise = how much changes" rule only applies to the genuine img2img passes
(upscale/refiner stages). Up to ~10 reference images can be chained (use the
`Image Stitch` node instead of chaining at 3+ refs — it's more reliable).

**Fact 2 — FLUX denoise behaves nothing like SD.** In real img2img, SD restyles
at denoise 0.4–0.6; FLUX barely changes below ~0.80 and needs ~0.85–0.95 for a
visible restyle. This is why our workflows use `ReferenceLatent` editing for
restyles and reserve low denoise (0.15–0.3) for detail-only refiner/upscale
passes.

Klein licensing: **4B is Apache-2.0** (commercial OK); **9B is non-commercial**.
Relevant if outputs are shipped commercially.

---

# Topic 1 — Detail refinement & high resolution

## 1.1 What the reference workflows do

| Workflow | Role | Core mechanism |
|---|---|---|
| [`Moody F2K Edit Workflow - V3.json`](../user/default/workflows/references/Moody%20F2K%20Edit%20Workflow%20-%20V3.json) | **The kitchen sink — best reference** | UltimateSDUpscale → skin-contrast blend → auto+manual detailers → SeedVR2 4K |
| [`flux2KleinRefiner_v21.json`](../user/default/workflows/references/flux2KleinRefiner_v21.json) | Generate → refiner → region detail | refiner img2img **denoise 0.15** + FaceDetailer regions @0.45 |
| [`PornMaster…Nipple_&_Areola Fix.json`](../user/default/workflows/references/PornMaster_F2K_9B_turbo_Nipple_&_Areola%20Fix_2026_05_27.json) | Cleanest **masked region-detailer** pattern | paint mask → `MaskToSEGS` → `DetailerForEach` **@0.6** + region LoRA |
| [`FLUX.2+Dev｜PiD直出4K.json`](../user/default/workflows/references/FLUX.2+Dev｜PiD直出4K.json) | **Direct 1024→4K** | NVIDIA PiD generative pixel-decode upscale |
| [`flux2Klein9BReference_v10.json`](../user/default/workflows/references/flux2Klein9BReference_v10.json) | Head/face swap | dual `ReferenceLatent` + `LanPaint_KSampler` inpaint |

The **Moody F2K Edit** workflow encodes the full canonical detail pipeline, all
stages toggle-able:

1. **Tiled diffusion upscale** — `UltimateSDUpscale`: upscale **2×**, tile
   **1024²**, **denoise 0.18**, **3 steps**, cfg 1, `euler`/`beta`, **Chess** mode,
   padding 64, paired with ESRGAN `4x-ClearRealityV1.pth`. Low denoise = refine,
   not invent. (Author's alt upscalers: `4xNomosWebPhoto_RealPLKSR` balanced,
   `4xNomos8k_atd_jpg` best-but-slow.)
2. **Skin micro-texture** — a **1× model** `1xSkinContrast-High-SuperUltraCompact.pth`
   via `ImageUpscaleWithModel`, blended back at **0.2–0.4 opacity** (`ImageBlend`).
   No resolution change, just pore/contrast detail. Off by default
   ("for super-realistic Western/photoreal").
3. **Localized detailers** — `UltralyticsDetectorProvider` (`face_yolov8m.pt`) →
   `BboxDetectorSEGS` → `ImpactSEGSOrderedFilter` → `DetailerForEach` auto face
   **@denoise 0.4**, plus a manual mask-painted `DetailerForEach` **@0.5
   (push to 0.54** to let a character LoRA take over the face). `ModelSamplingAuraFlow`
   `shift 3` is applied before detailer sampling.
4. **True 4K** — `SeedVR2VideoUpscaler` (DiT `seedvr2_ema_7b_sharp_fp16`,
   **resolution 4096, batch 1, color-fix `lab`**).

**Author's method, worth adopting wholesale:** generate the base **without** a
character LoRA (describe the character in the prompt), then "face-swap" the
character LoRA in only at the final Detailer stage. This stops the LoRA's
identity from bleeding into the whole frame.

`flux2KleinRefiner_v21` shows a lighter pattern: a generate pass, then a refiner
**img2img on a different checkpoint at denoise 0.15** (lcm/beta57, 5 steps) to
bake in texture without changing composition, then automatic region `FaceDetailer`
passes (sam_vit_b + bbox detectors, denoise 0.45). `PornMaster` is the minimal
copyable region-detailer: paint mask → `MaskToSEGS` → `DetailerForEach`
(guide_size 600, denoise 0.6) with a region-specific detail LoRA + descriptive
prompt.

## 1.2 Technique menu (current best practice)

A 2026 head-to-head benchmark
([rik-python detailer/skin workflows](https://github.com/rik-python/Comfyu--Image-detailer-and-skin-detailer-workflows))
states the tool choice bluntly: **SeedVR2 = best all-around; Flux DyPE = best raw
detail; SRPO = best skin.**

### SeedVR2 — generative restoration upscaler (recommended default)
[`numz/ComfyUI-SeedVR2_VideoUpscaler`](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
— a ByteDance diffusion-transformer one-step restoration model. Unlike ESRGAN
(which only sharpens existing edges) it *re-synthesizes* plausible pores, fabric
weave, and hair. Works on single images (`batch_size 1`; the 4n+1 batch rule is
video-only). `resolution` = target **shortest edge**.

- Models: `seedvr2_ema_7b_sharp_fp16` (best, 24 GB+), `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16`
  hybrid (16–20 GB), `seedvr2_ema_3b_fp16` / `seedvr2_ema_3b-Q8_0.gguf` (8–12 GB).
  Avoid the plain 7B fp8 (known quality issues).
- 4 nodes: `SeedVR2 (Down)Load DiT Model`, `SeedVR2 (Down)Load VAE Model`,
  `SeedVR2 Torch Compile Settings`, `SeedVR2 Video Upscaler`. Auto-download to
  `models/SEEDVR2`.
- `color-fix lab` to keep colors faithful. BlockSwap (`blocks_to_swap`, offload
  cpu) for low VRAM — **disabled on Apple Silicon**. VAE tiling (tile 1024 /
  overlap 128) is the real bottleneck.

### UltimateSDUpscale — tiled diffusion (prompt-guided 2× control)
[`ssitu/ComfyUI_UltimateSDUpscale`](https://github.com/ssitu/ComfyUI_UltimateSDUpscale)
— denoise is *the* knob: **0.15–0.35 = detail-preserving** (our workflows sit at
0.15–0.25), 0.35–0.5 = more invention. Tile 512 safe / 1024 if VRAM allows.
Seam fix: "Half Tile + Intersections". Pair with `4x-UltraSharp` (crisp/anime),
`4x_NMKD-Siax_200k` (photo), or `RealESRGAN_x4plus` (general).

**Kill the duplicated-detail failure mode:** keep denoise ≤0.3, add a
**ControlNet Tile ~0.6** to lock structure, use 256 overlap + Half-Tile seam fix,
and keep prompts *generic* ("detailed skin, photorealistic" — exactly what our
upscale prompts do) rather than naming objects (named objects replicate per-tile).

### Impact Pack detailers — localized face/hand/eye detail
`UltralyticsDetectorProvider` (bbox: `face_yolov8m.pt`, `hand_yolov8s.pt`; seg:
`person_yolov8m-seg.pt`) → SEGS → `FaceDetailer` / `DetailerForEach`, with
`SAMLoader` (`sam_vit_b_01ec64.pth`) to tighten masks.

- `guide_size` **512–768** is what actually adds detail (the crop is upscaled to
  this before re-diffusing); `max_size` 1024–1500; `crop_factor` ~3; `feather`
  high to hide paste seams; `force_inpaint` on.
- **denoise 0.3–0.5 for faces** (≤0.4 stays identity-safe; 0.54 lets a character
  LoRA dominate). Two-pass (structural then a low-denoise 0.3 polish) is the
  portrait pro move.
- All these detectors plus `sam_vit_b` are already present in this repo's
  `models/ultralytics` and `models/sams`.

### Newer detail injectors worth adding
- **Detail Daemon** ([`Jonseed/ComfyUI-Detail-Daemon`](https://github.com/Jonseed/ComfyUI-Detail-Daemon)) —
  manipulates sigmas to inject texture while preserving composition. For **Flux
  use `detail_amount` 0.3–0.6** (SDXL needs <0.25). `start` 0.1–0.5, `end` 0.5–0.9.
- **PiD = NVIDIA Pixel Diffusion** ([`nv-tlabs/PiD`](https://github.com/nv-tlabs/PiD);
  ComfyUI ports [Merserk](https://github.com/Merserk/ComfyUI-PiD) /
  [tsolful](https://github.com/tsolful/ComfyUI-PiD)) — this is what
  `FLUX.2+Dev｜PiD直出4K` uses. It is **not** "progressive detail"; it's a
  latent-conditioned pixel-space diffusion *decoder* that super-resolves in one
  generative pass (`LATENT + caption + sigma → IMAGE`). Checkpoints `2k`
  (512→2048) and `2kto4k` (1024→4096); **backbones natively include
  `flux2-klein-9b`**. Settings `pid_steps 4, cfg 1.0, strength ~0.4`. In our
  workflow: base render at 1024, captured latent+sigma from a 50-step sampler
  (capture_step 45) → `PiDPrepare(2kto4k, scale 4)` → `PiDSample(4)` → `PiDFinalize`.
- **DyPE** ([`wildminder/ComfyUI-DyPE`](https://github.com/wildminder/ComfyUI-DyPE),
  ICML 2026) — tweaks the DiT RoPE + noise schedule to generate *natively* at
  4K/8K in one coherent pass (no tile seams). Mostly txt2img; steps 28–50, cfg
  3.5–5; pairs with PiD as a finisher.
- **SRPO** (Tencent) — a FLUX.1-dev finetune built to kill the CGI/plastic-skin
  look (adds pores, micro-reflections). Used as a low-denoise refiner *after*
  SeedVR2; **SeedVR2 → SRPO is rated best-for-skin**, beating SUPIR (which
  airbrushes).

### Detail/texture LoRAs
Stack into the refiner/detailer pass. Standout for our stack:
**"Ultra Real - Klein 9B"** ([civitai 2462105](https://civitai.com/models/2462105/ultra-real-klein-9b))
— built to de-plasticize Klein 9B; weight **0.55 editing / 0.7–0.8 generation**;
triggers "High-quality photograph of a" / "Make this photo high-quality". The
Moody workflow already uses `Flux2-Klein-9B-consistency-V2 @0.54` similarly.

### De-AI finishing (breaks the plastic look)
- **CAS sharpen** — `ImageContrastAdaptiveSharpening` (in our installed
  `ComfyUI_essentials`) — micro-contrast without halos.
- **Film grain** sized to output resolution — the single most effective
  de-plastic finisher (`comfyui-propost`, `ComfyUI-Image-Filters`).
- **Color-match to source** in LAB space after any high-denoise pass.

## 1.3 Recommended detail/4K recipe

```
Input
 → SeedVR2  (7b_sharp_fp16 if 24GB+, else 3b_fp16 / 3b-Q8 gguf;
             resolution = target short edge: 1440 for 2K, 2160 for 4K; color-fix lab)
 → Klein 9B low-denoise refine  (ReferenceLatent anchors identity; denoise 0.20–0.30;
             + Ultra-Real-Klein-9B LoRA @0.55; + Detail Daemon detail_amount ~0.4)
 → DetailerForEach on face_yolov8m (+ sam_vit_b)  (guide_size 768, denoise 0.35–0.4, crop_factor 3)
 → CAS sharpen + film grain + color-match-to-input
```

- Swap in **UltimateSDUpscale @denoise 0.18 + ControlNet Tile** when you want
  tile-level prompt-guided control at 2×.
- Use **PiD `2kto4k`** for a fast single-pass 1024→4K finish on Klein.
- **Guardrails:** keep the refiner ≤0.35 (over-denoise wanders off identity);
  never name specific objects in tile/upscale prompts (they replicate per-tile).

---

# Topic 2 — Style conversion / repaint

There are **four mechanisms**, ordered by how hard they lock composition. Best
results stack them. Our workflows lean heavily on the first two.

## 2.1 What the reference workflows do

| Workflow | Direction | Mechanism |
|---|---|---|
| [`Moody Anime2Real V3.2.json`](../user/default/workflows/references/Moody%20Anime2Real%20V3.2.json) | **anime/2D → photoreal** | Klein reference-edit + caption-rewrite, then **Z-Image img2img @0.54** |
| [`FLUX2_Img2Img_Workflow_v777-secret.json`](../user/default/workflows/references/FLUX2_Img2Img_Workflow_v777-secret.json) | general repaint / 3D multiview | Klein `ReferenceLatent` edit + Anything2Real LoRAs |
| [`DA_flux2_klein-9b_distilled_union_v5.json`](../user/default/workflows/references/DA_flux2_klein-9b_distilled_union_v5.json) | template (txt2img/img2img/edit/inpaint/outpaint) | multi-mode Klein `ReferenceLatent` (up to 5 refs) |
| [`Flux 2D & Klein_9b ver 5.0.3.json`](../user/default/workflows/references/Flux%202D%20&%20Klein_9b%20ver%205.0.3.json) | photoreal editor | FLUX.2-dev↔Klein switch + 5× `ReferenceLatent` + photoreal FX stack |

Note: **no reference workflow does the realistic→2D direction.** "Flux 2D" means
a FLUX.2-**dev**↔Klein model switch (not anime); its SDXL/Pony checkpoint
(`gonzalomoXLFluxPony`) is used **only for the detailer passes**, and its 2D-style
LoRA slots are empty with the author note "nobody has created Flux2 Klein 2D
LoRAs yet." Section 2.5 gives the mirror recipe to fill that gap.

## 2.2 Mechanism 1 — Klein/Kontext instruction editing (best identity/composition lock)

Graph (the heart of `FLUX2_Img2Img`, `DA_union`, `Flux 2D`, and Moody stage 1):

```
LoadImage → resize (multiple of 16) → VAEEncode → ReferenceLatent ─┐
CLIPTextEncode(instruction) → FluxGuidance ───────────────────────┴→ KSampler(denoise 1.0) → VAEDecode
```

- Negative = `ConditioningZeroOut` of the positive (Klein has no real negative).
- **Distilled Klein:** 4 steps, cfg 1, `euler`/`simple`. **Base Klein:** 20 steps,
  cfg 5. **FLUX.1 Kontext:** `FluxGuidance` 2.5, 20–28 steps.
- **FluxGuidance with reference images: use ~1.0 or 3.5; high guidance destroys
  consistency** with the reference (per the Flux 2D author note).

**Writing the instruction is the skill.** Be imperative, name the target medium,
pin what must NOT change:

- "Convert this anime drawing into a photorealistic photograph. Keep the exact
  same pose, composition, framing, and facial identity. Realistic skin texture,
  natural studio lighting."
- "Restyle into a 2D anime illustration, clean cel shading, bold lineart.
  Preserve pose, expression, clothing, and background layout."
- "Turn this 3D render into a hand-drawn 2D illustration, flat colors, visible
  ink outlines. Same camera angle and proportions."
- "Convert this 2.5D character into a realistic 3D render, PBR materials, soft
  global illumination. Keep identical pose."

**Documented limitation:** Klein/Kontext is great at *text-described* restyle of
one image, but unreliable at "copy the style *from* image B onto image A" (it
tends to output them side-by-side). For that, verbalize image B with a VLM into
the prompt, use a trained conversion LoRA, or use IP-Adapter/Redux.

### The anime→real trick in Moody Anime2Real (genuinely good, transferable)

`Moody Anime2Real V3.2` is a **two-stage cascade across two model families**:

**Stage 1 — Klein reference-edit** on `Moody-Desire-V1_fp8` (FLUX.2 Klein 9B) with
the **`Flux2 Klein_Anything to Real Characters` LoRA @1.0**. Instruction prompt:
*"Realistic style of a young Asian girl. Remove watermarks."* — 3 steps,
`euler`/`simple`, cfg 1.

**Stage 2 — Z-Image img2img @ denoise 0.54** on `moody-real-v5` (a Z-Image model,
`qwen_3_4b` type `lumina2`, `ae.safetensors`), prompted by an auto-built caption:

1. `Florence2Run` (`MiaoshouAI/Florence-2-base-PromptGen-v2.0`, task
   **`prompt_gen_mixed_caption`**) captions the *anime* input.
2. A **linear chain of `StringReplace` nodes rewrites the caption's medium words**
   (verified order): `digital illustration→realistic photography`,
   `Anime/anime→Realistic/realistic`, `animated→realistic`, `drawing→photo`,
   `art style→style`, `illustration→photography`, strip `watermark` /
   `artist's signature`. The model's own description of an anime image becomes a
   *photo* prompt.
3. Concatenated with a manual photo prefix ("19-yo beauty, ultra cool-white skin,
   fine pores & real fuzz visible, 8K, real portrait photography") → `CLIPTextEncode`
   → the **denoise 0.54** sampler (keeps composition, pushes rendering to photoreal).

Then `UltimateSDUpscale` (denoise 0.25), `FaceDetailer` (0.45, the character
face-swap stage), a `1xSkinContrast` overlay (0.4), and optional SeedVR2 4K.

`FLUX2_Img2Img` is the same Klein reference-edit idea in API form, with a
**`CR LoRA Stack`** of `klein_9B_Turbo_r128 @1.0` + `f2k_anything2real_a_patched
@0.4` + `A2R_Klein_Standard @0.4`. Its notes also show a **2D→3D multiview** use
("generate front/side/back orthographic references… rendered 3D model style,
maximum consistency").

## 2.3 Mechanism 2 — Conversion LoRAs (highest-leverage add-on)

The single biggest quality lever for style flips. Center of gravity is **FLUX.2
Klein 9B-base**, with an official training guide
([Fine-tune Klein with a LoRA in <60 min](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora),
Jun 2026; [Civitai recipe](https://developer.civitai.com/orchestration/recipes/training-flux2-klein)).

On-target for our stack:
- **F2K 9B Anything2Real** (lrzjason, [civitai 2121900](https://civitai.com/models/2121900/flux2klein-9b-anything2real-lrzjason))
  — any style → photoreal; trigger *"transform the image into high quality
  realistic photograph. {male/female}"*; strength **0.8–1.0** (drop toward 0.5 if
  over-smoothing). Same family our workflows already use.
- **Consistence Edit LoRA Klein 9B** ([civitai 1939453](https://civitai.com/models/1939453/consistence-edit-lora)).
- **AniEdit (Flux 2 Klein) 9B** ([civitai 2332320](https://civitai.com/models/2332320/aniedit-flux-2-klein))
  — anime style transfer.
- Aesthetic Klein 9B LoRAs: Hand-Painted Anime Background, Retro Anime Texture,
  Portrait Engine (detailed skin) — on Civitai's "Flux.2 Klein LoRA" filter.

If dropping to **FLUX.1 Kontext**, the catalog is far larger — e.g. the
[Owen777 Kontext-Style collection](https://huggingface.co/Owen777/Kontext-Style-Loras)
(22 styles incl. Ghibli, 3D_Chibi, Clay_Toy, Pixel, LEGO; universal trigger
*"Turn this image into the [STYLE] style."*), **HXHY-RealisticKontextLoRA**
(anime→real, *"Convert to a realistic art style"*), **Game2Reality** (3D game→photo),
**gokaygokay Low-Poly** (→low-poly 3D).

**Critical loading gotcha:** FLUX stores attention as a fused QKV matrix while
community LoRAs ship in diffusers format (separate to_q/to_k/to_v), so the stock
`LoraLoaderModelOnly` **silently mis-applies most Klein LoRA weights**. Use the
architecture-aware [`Comfyui-flux2klein-Lora-loader`](https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader)
node for Klein 9B (this is why our `FLUX2_Img2Img` LoRA is named `..._patched`).

## 2.4 Mechanism 3 — ControlNet structure-lock (hard composition lock + heavy restyle)

ControlNet holds geometry *independently of denoise*, so you can run a full
repaint (denoise 0.9–1.0 / txt2img) and keep the layout. Used in `Flux 2D & Klein`
via `DepthAnythingV2Preprocessor` + HED.

Preprocessor by what you preserve (`comfyui_controlnet_aux`, installed):
- **Depth** (`DepthAnythingV2Preprocessor`, `depth_anything_v2_vitl.pth`) — loose
  geometry, max repaint freedom → **best default for cross-medium** (anime↔real,
  3D↔2D).
- **Lineart** (`AnimeLineArtPreprocessor` / `Manga2Anime_LineArt_Preprocessor`) —
  preserves contours → best for real→anime.
- **Canny** (`CannyEdgePreprocessor`) — hardest edge lock.
- **Pose** (`DWPreprocessor`) — locks character pose only.

Apply with **`ControlNetApplyAdvanced`** (its optional `vae` input handles Flux;
the old `ControlNetApplySD3` / "Apply Controlnet with VAE" is deprecated).
**FLUX strengths run lower than SD: 0.4–0.8**, and **ending control early
(`end_percent` 0.5–0.8) frees late steps to apply style** — the key
structure-then-style lever.

**FLUX.2 ControlNet status:** BFL ships none (they argue multi-reference handles
it). The only option is the community
[`alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
(canny/hed/depth/pose/mlsd/scribble/gray, scale **0.65–0.80**) — run via our own
`comfyui-flux2fun-controlnet` fork (which is exactly why it's in the manifest).
For FLUX.1-dev, the mature pick is **Shakker-Labs Union-Pro-2.0** (depth 0.8@end0.8,
canny 0.7@end0.8); load `ControlNetLoader` → `SetUnionControlNetType` →
`ControlNetApplyAdvanced`.

## 2.5 Mechanism 4 — img2img denoise & reference-style (Redux)

Straight img2img is the crudest restyle, and on FLUX needs denoise 0.85–0.95 to
do anything (Fact 2). For style-*reference* (borrow a look from an image) FLUX uses
**Redux** (`StyleModelApply` + `flux1-redux-dev` + `sigclip_vision_patch14_384`);
lower its strength / raise `ReduxAdvanced` `downsampling_factor` (3–5) so the text
prompt still matters. IP-Adapter (`IPAdapterAdvanced`, weight_type "style transfer")
is SD/SDXL-only — not FLUX.

## 2.6 Recommended recipes per direction

- **Anime/2D → realistic:** Klein 9B + **Anything2Real LoRA @0.8–1.0** + the
  Florence2-caption→`StringReplace` trick (§2.2) → upscale → FaceDetailer @0.4.
  Add Depth ControlNet @0.5–0.6 (end 0.7) only if composition drifts.
- **Realistic → anime/2D (not in our refs — mirror recipe):** Klein instruction
  *"restyle into 2D anime, cel shading, keep pose/composition"* + an anime Klein
  LoRA at higher strength; or for a strong flat look, an SDXL/Pony anime checkpoint
  + **Lineart ControlNet @0.5–0.7**. Invert the `StringReplace` pairs
  (`photo→drawing`, `realistic→anime`).
- **3D render → 2D illustration:** clean render geometry makes ControlNet reliable
  — Depth+Lineart into an illustration checkpoint, or Klein *"convert to flat 2D
  illustration, ink outlines, same camera."*
- **2.5D → 3D:** Klein **base** (not distilled) *"realistic 3D render, PBR
  materials, soft GI, identical pose"* + Depth ControlNet to hold volume.

## 2.7 The governing mental model

Two antagonistic forces:

- **Structure-lock** (raise to preserve): lower denoise, higher ControlNet
  strength + later `end_percent`, `ReferenceLatent` editing, pose ControlNet.
- **Style-push** (raise to convert harder): higher denoise, higher LoRA strength,
  stronger/explicit instruction, stronger style model weight.

Raise one, lower the other — never max both. Tune order: denoise → ControlNet →
LoRA/reference. Concrete fixes: identity drift → drop denoise / add Depth+pose CN
/ lower LoRA / use a reference-edit. Style too weak → raise denoise(0.7+) or LoRA
(0.9–1.2) / use a dedicated conversion LoRA / use base (not distilled) Klein.
Washed composition → add Depth CN / raise its `end_percent`.

---

# 3. Auto-captioning to drive a repaint

Captioning makes a workflow input-agnostic: it reads the source and writes the
prompt so the restyle keeps the subject's semantics.

- **Florence2** ([`kijai/ComfyUI-Florence2`](https://github.com/kijai/ComfyUI-Florence2)):
  `DownloadAndLoadFlorence2Model` → `Florence2Run`. Use task **`more_detailed_caption`**
  (or `prompt_gen_mixed_caption` with the MiaoshouAI PromptGen models, as our
  workflows do) for restyle — the verbose prose carries the most subject detail and
  suits FLUX's Qwen/T5 encoders. Wire the caption STRING through `Text Concatenate`
  (prepend the style trigger) into `CLIPTextEncode.text`.
- **WD14 tagger** (`WD14Tagger|pysssss`, `wd-eva02-large-tagger-v3`) — comma tags;
  use only when the *target* is an SDXL anime/Pony/Illustrious checkpoint (booru
  prompting). `threshold` 0.35.
- The `StringReplace` medium-swap chain (§2.2) sits between the caption and the
  encode — that's the mechanical core of the anime→real flip.

---

# 4. Install gaps (run station)

To run these references beyond the Klein models, the run station needs custom
nodes not in [`custom_nodes.manifest.json`](../custom_nodes.manifest.json):
**SeedVR2, rgthree (Power Lora Loader), ComfyUI-Florence2, KJNodes,
ComfyUI-easy-use, LoraManager, Image Saver**, the
**Comfyui-flux2klein-Lora-loader** (reliable Klein LoRA loading), and optionally
**ComfyUI-PiD / ComfyUI-DyPE / ComfyUI-Detail-Daemon**. The `DA_StyleSelector`
preset text is not in the workflow JSON — it lives in the external
`DemonAlone-StyleSelector-ComfyUI` node's database.

Per AGENTS.md, custom-node changes go through the manifest/installer (and forks
for source fixes), and the installer is run by the user, not by the agent.

This repo *does* already have the detail-refinement essentials locally: Impact
Pack + Subpack, `comfyui_controlnet_aux`, `comfyui_face_parsing`,
`ComfyUI_essentials`, the `face_yolov8m` / `hand_yolov8s` / `person_yolov8m-seg`
detectors, `sam_vit_b`, and the `comfyui-flux2fun-controlnet` fork (FLUX.2
ControlNet).

---

# 5. Sources

**Models / official**
- FLUX.2 announcement & tiers — https://bfl.ai/blog/flux-2
- FLUX.2 Klein (ComfyUI) — https://docs.comfy.org/tutorials/flux/flux-2-klein ·
  https://comfy.org/workflows/model/flux-2-klein/
- Klein LoRA training — https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora
- FLUX.1 Kontext (ComfyUI) — https://docs.comfy.org/tutorials/flux/flux-1-kontext-dev
- `ReferenceLatent` / `FluxKontextImageScale` nodes —
  https://docs.comfy.org/built-in-nodes/ReferenceLatent

**Detail / upscale**
- SeedVR2 — https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
- UltimateSDUpscale — https://github.com/ssitu/ComfyUI_UltimateSDUpscale
- Impact Pack (FaceDetailer) — https://github.com/ltdrdata/ComfyUI-Impact-Pack
- Detail Daemon — https://github.com/Jonseed/ComfyUI-Detail-Daemon
- NVIDIA PiD — https://github.com/nv-tlabs/PiD · https://github.com/Merserk/ComfyUI-PiD
- DyPE — https://github.com/wildminder/ComfyUI-DyPE
- SRPO — https://github.com/Tencent-Hunyuan/SRPO
- Ultra Real Klein 9B LoRA — https://civitai.com/models/2462105/ultra-real-klein-9b
- Detailer/skin benchmark — https://github.com/rik-python/Comfyu--Image-detailer-and-skin-detailer-workflows

**Style conversion**
- Klein LoRA loader (fused-QKV fix) — https://github.com/capitan01R/Comfyui-flux2klein-Lora-loader
- Anything2Real Klein 9B — https://civitai.com/models/2121900/flux2klein-9b-anything2real-lrzjason
- Kontext style collection — https://huggingface.co/Owen777/Kontext-Style-Loras
- ControlNet aux preprocessors — https://github.com/Fannovel16/comfyui_controlnet_aux
- FLUX.2 community ControlNet — https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union
- FLUX.1-dev ControlNet Union-Pro-2.0 — https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0
- Florence2 — https://github.com/kijai/ComfyUI-Florence2
- Style transfer overview — https://blog.comfy.org/p/the-complete-style-transfer-handbook
