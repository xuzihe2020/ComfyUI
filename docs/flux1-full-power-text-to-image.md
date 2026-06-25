# Full-power FLUX.1 text-to-image workflow for ComfyUI

This is the proper FLUX.1-style text-to-image workflow: separate diffusion model, dual text encoders, explicit Flux guidance, explicit guider/scheduler/sampler stack, VAE decode, and save.

Use this when you want to understand and control FLUX properly. The simpler `KSampler` workflow can run, but it hides the Flux-specific sampling pieces that matter.

## Target model family

This workflow is for local FLUX.1 diffusion models such as:

- `flux1-dev.safetensors`
- `flux1-schnell.safetensors`
- `flux1-krea-dev.safetensors`

For "full power" quality, start with `flux1-dev.safetensors` or `flux1-krea-dev.safetensors`, not the single-file FP8 checkpoint shortcut.

## Required model files

Put the model files here, or expose equivalent folders through `extra_model_paths.yaml`.

```text
ComfyUI/
  models/
    diffusion_models/
      flux1-dev.safetensors

    text_encoders/
      clip_l.safetensors
      t5xxl_fp16.safetensors

    vae/
      ae.safetensors
```

For lower VRAM, replace `t5xxl_fp16.safetensors` with `t5xxl_fp8_e4m3fn.safetensors`. On an RTX 4090, prefer `t5xxl_fp16.safetensors` first.

## Nodes

Use these exact ComfyUI node classes:

| Purpose | Node class | UI display name |
|---|---|---|
| Load Flux diffusion model | `UNETLoader` | `Load Diffusion Model` |
| Load CLIP-L + T5-XXL text encoders | `DualCLIPLoader` | `Load CLIP (Dual)` |
| Patch Flux sampling behavior for resolution | `ModelSamplingFlux` | `ModelSamplingFlux` |
| Encode prompt | `CLIPTextEncode` | `CLIP Text Encode (Prompt)` |
| Apply Flux guidance strength | `FluxGuidance` | `FluxGuidance` |
| Convert model + conditioning into a guider | `BasicGuider` | `Basic Guider` |
| Select sampler algorithm | `KSamplerSelect` | `KSamplerSelect` |
| Create step/sigma schedule | `BasicScheduler` | `BasicScheduler` |
| Create noise from seed | `RandomNoise` | `RandomNoise` |
| Create empty Flux latent | `EmptySD3LatentImage` | `EmptySD3LatentImage` |
| Run the explicit sampler | `SamplerCustomAdvanced` | `SamplerCustomAdvanced` |
| Load Flux VAE | `VAELoader` | `Load VAE` |
| Decode latent to pixels | `VAEDecode` | `VAE Decode` |
| Save output | `SaveImage` | `Save Image` |
| Shared width value | `PrimitiveInt` | `Int` |
| Shared height value | `PrimitiveInt` | `Int` |

## High-quality graph diagram

```mermaid
flowchart LR
  %% -----------------------------
  %% Model loading lane
  %% -----------------------------
  subgraph M["1. Model loading"]
    direction TB
    UNET["UNETLoader<br/>Load Diffusion Model<br/><br/>unet_name: flux1-dev.safetensors<br/>weight_dtype: default or fp8_e4m3fn"]
    MSF["ModelSamplingFlux<br/><br/>max_shift: model-specific<br/>base_shift: model-specific<br/>width/height: from shared INT nodes"]
    DCLIP["DualCLIPLoader<br/>Load CLIP (Dual)<br/><br/>clip_name1: clip_l.safetensors<br/>clip_name2: t5xxl_fp16.safetensors<br/>type: flux"]
    VAELOAD["VAELoader<br/>Load VAE<br/><br/>vae_name: ae.safetensors"]
  end

  %% -----------------------------
  %% Shared parameter lane
  %% -----------------------------
  subgraph P["0. Shared parameters"]
    direction TB
    WIDTH["PrimitiveInt<br/>Int<br/><br/>value: 1024"]
    HEIGHT["PrimitiveInt<br/>Int<br/><br/>value: 1536 or 1024"]
  end

  %% -----------------------------
  %% Conditioning lane
  %% -----------------------------
  subgraph C["2. Prompt conditioning"]
    direction TB
    PROMPT["Prompt text<br/><br/>No SD-style negative prompt required<br/>for basic Flux.1 Dev"]
    ENCODE["CLIPTextEncode<br/>CLIP Text Encode (Prompt)"]
    FG["FluxGuidance<br/><br/>guidance: 3.5"]
    GUIDER["BasicGuider<br/><br/>MODEL + Flux-conditioned prompt"]
  end

  %% -----------------------------
  %% Sampling controls lane
  %% -----------------------------
  subgraph S["3. Sampling controls"]
    direction TB
    NOISE["RandomNoise<br/><br/>noise_seed: choose or randomize"]
    KSAMPLER["KSamplerSelect<br/><br/>sampler_name: euler"]
    SCHED["BasicScheduler<br/><br/>scheduler: simple<br/>steps: 20<br/>denoise: 1.0"]
    LATENT["EmptySD3LatentImage<br/><br/>width/height: from shared INT nodes<br/>batch_size: 1"]
  end

  %% -----------------------------
  %% Generation lane
  %% -----------------------------
  subgraph G["4. Generation and output"]
    direction TB
    ADV["SamplerCustomAdvanced<br/><br/>explicit Flux sampling"]
    DECODE["VAEDecode<br/>VAE Decode"]
    SAVE["SaveImage<br/><br/>filename_prefix: flux1/dev"]
  end

  DCLIP -- "CLIP" --> ENCODE
  PROMPT -- "text" --> ENCODE
  ENCODE -- "CONDITIONING" --> FG
  FG -- "CONDITIONING" --> GUIDER
  UNET -- "MODEL" --> MSF
  WIDTH -- "INT width" --> MSF
  HEIGHT -- "INT height" --> MSF
  MSF -- "patched MODEL" --> GUIDER

  MSF -- "patched MODEL" --> SCHED
  WIDTH -- "INT width" --> LATENT
  HEIGHT -- "INT height" --> LATENT
  NOISE -- "NOISE" --> ADV
  GUIDER -- "GUIDER" --> ADV
  KSAMPLER -- "SAMPLER" --> ADV
  SCHED -- "SIGMAS" --> ADV
  LATENT -- "LATENT" --> ADV

  ADV -- "output LATENT" --> DECODE
  VAELOAD -- "VAE" --> DECODE
  DECODE -- "IMAGE" --> SAVE
```

## Exact wiring checklist

Wire the nodes like this:

```text
DualCLIPLoader.CLIP
  -> CLIPTextEncode.clip

Prompt text
  -> CLIPTextEncode.text

CLIPTextEncode.CONDITIONING
  -> FluxGuidance.conditioning

FluxGuidance.CONDITIONING
  -> BasicGuider.conditioning

UNETLoader.MODEL
  -> ModelSamplingFlux.model

PrimitiveInt width.INT
  -> ModelSamplingFlux.width
  -> EmptySD3LatentImage.width

PrimitiveInt height.INT
  -> ModelSamplingFlux.height
  -> EmptySD3LatentImage.height

ModelSamplingFlux.MODEL
  -> BasicGuider.model

ModelSamplingFlux.MODEL
  -> BasicScheduler.model

RandomNoise.NOISE
  -> SamplerCustomAdvanced.noise

BasicGuider.GUIDER
  -> SamplerCustomAdvanced.guider

KSamplerSelect.SAMPLER
  -> SamplerCustomAdvanced.sampler

BasicScheduler.SIGMAS
  -> SamplerCustomAdvanced.sigmas

EmptySD3LatentImage.LATENT
  -> SamplerCustomAdvanced.latent_image

SamplerCustomAdvanced.output
  -> VAEDecode.samples

VAELoader.VAE
  -> VAEDecode.vae

VAEDecode.IMAGE
  -> SaveImage.images
```

## Recommended starting settings for FLUX.1 Dev

```text
UNETLoader:
  unet_name: flux1-dev.safetensors
  weight_dtype: default

PrimitiveInt width:
  value: 1024

PrimitiveInt height:
  value: 1024

ModelSamplingFlux:
  max_shift: use the model creator's recommendation
  base_shift: use the model creator's recommendation
  width: connect from PrimitiveInt width
  height: connect from PrimitiveInt height

DualCLIPLoader:
  clip_name1: clip_l.safetensors
  clip_name2: t5xxl_fp16.safetensors
  type: flux

VAELoader:
  vae_name: ae.safetensors

FluxGuidance:
  guidance: 3.5

BasicScheduler:
  scheduler: simple
  steps: 20
  denoise: 1.0

KSamplerSelect:
  sampler_name: euler

EmptySD3LatentImage:
  width: connect from PrimitiveInt width
  height: connect from PrimitiveInt height
  batch_size: 1

SaveImage:
  filename_prefix: flux1/dev
```

## Notes that matter

- `UNETLoader` is badly named for Flux. In the UI it appears as `Load Diffusion Model`. It loads the main Flux diffusion transformer model, even though Flux is not a classic SD UNet.
- `ModelSamplingFlux` should sit after `UNETLoader` in a full-control Flux graph. Its width/height should match the actual latent width/height.
- Use shared `PrimitiveInt` nodes for width and height. Connect them to both `ModelSamplingFlux` and `EmptySD3LatentImage` so the resolution cannot drift out of sync.
- `FluxGuidance` is the Flux-specific guidance node. For FLUX.1 Dev, `3.5` is a strong default.
- Do not think in SD negative-prompt terms first. Basic Flux.1 text-to-image usually uses one positive prompt plus `FluxGuidance`.
- `SamplerCustomAdvanced` is preferred here because it makes the Flux-native sampler stack explicit: noise, guider, sampler, sigmas, latent.
- `KSampler` can run some Flux workflows, but it hides too much and is not the graph to study if your goal is real understanding.
- The VAE is `ae.safetensors`, not a normal SD VAE.

## Minimal mental model

```text
Flux model -> ModelSamplingFlux + CLIP-L/T5 prompt + FluxGuidance
        -> BasicGuider
        -> SamplerCustomAdvanced with noise/sampler/sigmas/latent
        -> VAE Decode with ae.safetensors
        -> Save Image
```
