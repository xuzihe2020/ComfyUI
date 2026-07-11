#!/usr/bin/env bash
# First-session pod bootstrap. Idempotent — safe to re-run, skips what exists.
# Process definition: skills/runpod-comfyui-setup.md. Do not deviate from it.
#
# ORDER MATTERS: all cloning happens BEFORE any heavy dependency install, so
# the workspace is visibly complete early and a dead pod mid-bootstrap leaves
# resumable state. Dependencies converge at the end.
set -uo pipefail
# Never hang on a credential prompt (private repos fail fast and visibly).
export GIT_TERMINAL_PROMPT=0

FORK=https://github.com/xuzihe2020/ComfyUI
COMFY=/workspace/ComfyUI
AITK=/workspace/ai-toolkit

echo "=== [1/9] ComfyUI: Tony's fork is the deployment unit ==="
if [ -d "$COMFY/.git" ]; then
  origin=$(git -C "$COMFY" remote get-url origin)
  if [ "$origin" != "$FORK" ] && [ "$origin" != "$FORK.git" ]; then
    echo "!! $COMFY is cloned from $origin, not the fork — replacing"
    rm -rf "$COMFY"
  fi
fi
if [ ! -d "$COMFY/.git" ]; then
  git clone "$FORK" "$COMFY" || exit 1
fi

echo "=== [2/9] Tony's forked custom nodes: direct clone from his GH (manifest-driven) ==="
python3 - <<'PY'
import json, subprocess
from pathlib import Path

root = Path("/workspace/ComfyUI")
manifest = json.loads((root / "custom_nodes.manifest.json").read_text(encoding="utf-8"))
for node in manifest["nodes"]:
    if not node["repo"].startswith("https://github.com/xuzihe2020/"):
        continue
    platforms = node.get("platforms")
    if platforms and "linux" not in platforms:
        continue
    target = root / "custom_nodes" / node["folder"]
    if target.exists():
        print(f"skip (exists): {target.name}", flush=True)
        continue
    subprocess.run(["git", "clone", node["repo"], str(target)], check=True)
PY

echo "=== [3/9] ai-toolkit: clone pinned fork ==="
if [ ! -d "$AITK/.git" ]; then
  git clone https://github.com/xuzihe2020/ai-toolkit "$AITK" || exit 1
fi

echo "=== [4/9] venvs (in-repo; inherit template torch via system-site-packages) ==="
if [ ! -d "$COMFY/.venv" ]; then
  python -m venv --system-site-packages "$COMFY/.venv" || exit 1
fi
if [ ! -d "$AITK/.venv" ]; then
  python -m venv --system-site-packages "$AITK/.venv" || exit 1
fi
"$COMFY/.venv/bin/python" -m pip install -q -U pip
"$AITK/.venv/bin/python" -m pip install -q -U pip

echo "=== [5/9] extra_model_paths.yaml -> /workspace/comfyui_models ==="
mkdir -p /workspace/comfyui_models
if [ ! -f "$COMFY/extra_model_paths.yaml" ]; then
  cat > "$COMFY/extra_model_paths.yaml" <<'EOY'
comfyui:
    base_path: /workspace/comfyui_models/
    is_default: true
    download_model_base: /workspace/comfyui_models/
    checkpoints: checkpoints/
    diffusion_models: |
        diffusion_models/
        unet/
    text_encoders: |
        text_encoders/
        clip/
    clip_vision: clip_vision/
    configs: configs/
    controlnet: controlnet/
    embeddings: embeddings/
    loras: loras/
    upscale_models: upscale_models/
    vae: vae/
EOY
  echo "wrote default extra_model_paths.yaml — verify folder names match the synced model tree"
fi

echo "=== [6/9] manager + community nodes + tools + node dependencies: install script ==="
(cd "$COMFY" && "$COMFY/.venv/bin/python" scripts/install_custom_nodes.py) || exit 1

echo "=== [7/9] ComfyUI requirements (the slow pip — deliberately after all cloning) ==="
"$COMFY/.venv/bin/python" -m pip install -r "$COMFY/requirements.txt" || exit 1

echo "=== [8/9] ai-toolkit requirements ==="
"$AITK/.venv/bin/python" -m pip install -r "$AITK/requirements.txt" || exit 1

echo "=== [9/9] workspace layout + volume tools + per-session sync script ==="
mkdir -p /workspace/{hf-cache,datasets,configs,output,scripts,rclone,bin,tmp}

# AWS CLI, persistent on the volume (model_sync transfers/maintenance need it;
# the pod image doesn't ship it). /workspace/bin is on PATH via session_start.
if [ ! -x /workspace/bin/aws ]; then
  cd /workspace/tmp \
    && curl -sL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip \
    && unzip -q awscliv2.zip \
    && ./aws/install --install-dir /workspace/aws-cli --bin-dir /workspace/bin --update >/dev/null \
    && rm -rf awscliv2.zip aws \
    && /workspace/bin/aws --version \
    || echo "!! aws cli install failed — model_sync maintenance/transfers won't run on pods until it's installed"
  cd /
fi

# session_start.sh is versioned in the fork; the volume copy self-updates on
# each sync (see scripts/runpod/session_start.sh).
if [ -f "$COMFY/scripts/runpod/session_start.sh" ]; then
  cp -f "$COMFY/scripts/runpod/session_start.sh" /workspace/scripts/session_start.sh
  chmod +x /workspace/scripts/session_start.sh
else
  echo "!! $COMFY/scripts/runpod/session_start.sh missing from the fork — scp it to /workspace/scripts/ manually"
fi

echo "=== torch sanity (must stay the template build in both venvs) ==="
"$COMFY/.venv/bin/python" - <<'PY'
import torch
print("comfy venv:", torch.__version__, "| cuda:", torch.cuda.is_available())
PY
"$AITK/.venv/bin/python" - <<'PY'
import torch
print("aitk venv:", torch.__version__, "| cuda:", torch.cuda.is_available())
PY

echo "BOOTSTRAP_DONE"
