#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — install ComfyUI + NVIDIA/PyTorch env + custom nodes on a RunPod
# pod. Idempotent: safe to re-run. Everything lives on the persistent
# /workspace network volume so it survives pod stop/recreate.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; source "$HERE/.env"; set +a; }

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY="${COMFY:-$WORKSPACE/ComfyUI}"
VENV="${VENV:-$WORKSPACE/venv}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

echo "### 1/6  system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git git-lfs tmux aria2 ffmpeg libgl1 libglib2.0-0 curl ca-certificates
git lfs install || true

echo "### 2/6  python venv -> $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

echo "### 3/6  PyTorch ($TORCH_CUDA)"
if ! python -c "import torch" 2>/dev/null; then
  pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

echo "### 4/6  ComfyUI -> $COMFY"
[ -d "$COMFY/.git" ] || git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY"
pip install -r "$COMFY/requirements.txt"

echo "### 5/6  custom nodes"
mkdir -p "$COMFY/custom_nodes"
cd "$COMFY/custom_nodes"
clone () { local d; d="$(basename "$1" .git)"; [ -d "$d" ] || git clone --depth 1 "$1"; }
clone https://github.com/ltdrdata/ComfyUI-Manager.git            # 1-click install any other missing nodes
clone https://github.com/bryanmcguire/comfyui-flux2fun-controlnet.git  # the FLUX.2 Fun ControlNet
clone https://github.com/Fannovel16/comfyui_controlnet_aux.git   # OpenPose/DWPose/Canny/Depth preprocessors
for d in */; do
  [ -f "${d}requirements.txt" ] && pip install -r "${d}requirements.txt" || true
done

echo "### 6/6  verify NVIDIA / torch / CUDA"
nvidia-smi || { echo "!! nvidia-smi failed — driver/GPU not visible"; exit 1; }
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to PyTorch — wrong wheel or driver mismatch"
print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
free, total = torch.cuda.mem_get_info()
print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
PY

echo
echo "OK. Next:"
echo "  1) cp runpod/.env.example runpod/.env  &&  edit HF_TOKEN"
echo "  2) bash runpod/download_models.sh"
echo "  3) bash runpod/start.sh"
