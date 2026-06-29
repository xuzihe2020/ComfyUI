#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh — provision a RunPod pod by cloning THIS ComfyUI repo and installing
# its environment + custom nodes through the repo's own manifest installer.
#
# Intended flow (you run this; it does not bypass repo conventions):
#   git clone $REPO_URL /workspace/ComfyUI
#   cd /workspace/ComfyUI && bash runpod/setup.sh
# (Running from a fresh pod without a clone also works — it will clone for you.)
#
# Idempotent. Everything lives on the /workspace network volume so it survives
# pod stop/recreate.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; source "$HERE/.env"; set +a; }

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/xuzihe2020/ComfyUI}"
COMFY="${COMFY:-$WORKSPACE/ComfyUI}"
VENV="${VENV:-$COMFY/.venv}"          # installer expects <repo-root>/.venv
TORCH_CUDA="${TORCH_CUDA:-cu124}"

echo "### 1/6  system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git git-lfs tmux aria2 ffmpeg libgl1 libglib2.0-0 curl ca-certificates
git lfs install || true

echo "### 2/6  clone/refresh this ComfyUI repo -> $COMFY"
if [ -d "$COMFY/.git" ]; then
  git -C "$COMFY" pull --ff-only || echo "  (pull skipped — local changes present)"
else
  git clone "$REPO_URL" "$COMFY"
fi

echo "### 3/6  venv at $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

echo "### 4/6  PyTorch ($TORCH_CUDA) + ComfyUI requirements"
python -c "import torch" 2>/dev/null || \
  pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install -r "$COMFY/requirements.txt"

echo "### 5/6  custom nodes via the repo's manifest installer"
# Installs everything in custom_nodes.manifest.json (incl. comfyui-flux2fun-controlnet
# and comfyui_controlnet_aux, whose native deps are force-installed via
# ALWAYS_FIX_DEPENDENCIES). This is the repo's own clone+install path.
cd "$COMFY"
python script/install_custom_nodes.py

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
echo "  1) cp runpod/.env.example runpod/.env  &&  edit HF_TOKEN  (if not done)"
echo "  2) bash runpod/download_models.sh"
echo "  3) bash runpod/start.sh"
