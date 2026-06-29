#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_models.sh — pull all FLUX.2-dev (BF16) + Alibaba ControlNet weights
# from HuggingFace into the ComfyUI model folders. Resumable & idempotent.
#
# Total download (BF16 set): ~108 GB
#   flux2-dev.safetensors ............... 64.4 GB  (diffusion_models)  [GATED]
#   mistral_3_small_flux2_bf16 .......... 35.6 GB  (text_encoders)
#   flux2-vae.safetensors ...............  0.3 GB  (vae)
#   FLUX.2-dev-Fun-Controlnet-Union .....  8.3 GB  (controlnet)
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; source "$HERE/.env"; set +a; }

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY="${COMFY:-$WORKSPACE/ComfyUI}"
VENV="${VENV:-$COMFY/.venv}"          # matches setup.sh / the repo installer

# Keep HF's download cache on the big /workspace volume, NOT the ~80 GB container
# disk (/). Otherwise the 64 GB transformer blob can fill / and abort the download.
export HF_HOME="${HF_HOME:-$WORKSPACE/.hf_cache}"
mkdir -p "$HF_HOME"

# shellcheck disable=SC1090
source "$VENV/bin/activate"
pip install -q -U "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1     # fast multi-threaded downloads

: "${HF_TOKEN:?ERROR: set HF_TOKEN in runpod/.env (accept the FLUX.2-dev license on HF first)}"
export HF_TOKEN

DM="$COMFY/models/diffusion_models"
TE="$COMFY/models/text_encoders"
VAE="$COMFY/models/vae"
CN="$COMFY/models/controlnet"
mkdir -p "$DM" "$TE" "$VAE" "$CN"

# hf download recreates the repo's path under --local-dir; flatten() moves the
# leaf file up to the target dir and removes the leftover split_files/ tree.
flatten () { local dir="$1" sub="$2"; mv -f "$dir/$sub/"*.safetensors "$dir/" && rm -rf "$dir/${sub%%/*}"; }

echo ">>> [1/4] FLUX.2-dev BF16 transformer (64.4 GB, gated)"
hf download black-forest-labs/FLUX.2-dev flux2-dev.safetensors --local-dir "$DM"

echo ">>> [2/4] Mistral text encoder BF16 (35.6 GB)"
hf download Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors --local-dir "$TE"
flatten "$TE" "split_files/text_encoders"

echo ">>> [3/4] FLUX.2 VAE (0.3 GB)"
hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors --local-dir "$VAE"
flatten "$VAE" "split_files/vae"

echo ">>> [4/4] Alibaba Fun ControlNet Union (8.3 GB)"
hf download alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union FLUX.2-dev-Fun-Controlnet-Union.safetensors --local-dir "$CN"

# --- OPTIONAL: FP8 set for a BF16-vs-FP8 quality A/B (uncomment to fetch) -----
# echo ">>> (opt) FP8-mixed transformer (35.5 GB)"
# hf download Comfy-Org/flux2-dev split_files/diffusion_models/flux2_dev_fp8mixed.safetensors --local-dir "$DM"
# flatten "$DM" "split_files/diffusion_models"
# echo ">>> (opt) Mistral text encoder FP8 (18 GB)"
# hf download Comfy-Org/flux2-dev split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors --local-dir "$TE"
# flatten "$TE" "split_files/text_encoders"

echo
echo "=== models on disk ==="
ls -lh "$DM" "$TE" "$VAE" "$CN"
df -h "$WORKSPACE"
