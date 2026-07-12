#!/usr/bin/env bash
# EVERY POD SPIN-UP — no exceptions: pull repos, pull every node, sync deps.
# Verdict-style output: every step prints [OK]/[FAIL]; run ends with a scoreboard.
# The manifest installer ALWAYS runs — it is self-skipping via its own state
# stamp (custom_nodes/.install_state.json): a no-change run exits in seconds.
#
# SINGLE SOURCE OF TRUTH: this file, in the fork. /workspace/scripts/
# session_start.sh is a SYMLINK to it — there is no copy to drift. Recreate:
#   ln -sfn /workspace/ComfyUI/scripts/runpod/session_start.sh /workspace/scripts/session_start.sh
# (One caveat of self-pulling: if a pull updates THIS file, the in-flight run
# may glitch once — just run it again; the next run is the new version.)
# Process definition: skills/runpod-comfyui-aitk-setup.md
set -uo pipefail
export HF_HOME=/workspace/hf-cache
export GIT_TERMINAL_PROMPT=0
# rclone: persistent binary + Google Drive config (pinned to Flux Prod —
# see skills/runpod-gdrive-rclone.md)
export PATH=/workspace/bin:$PATH
export RCLONE_CONFIG=/workspace/rclone/rclone.conf
chmod 600 "$RCLONE_CONFIG" 2>/dev/null || true  # S3 uploads don't carry file modes
COMFY=/workspace/ComfyUI
AITK=/workspace/ai-toolkit
FAILED=()
OK_COUNT=0

step_pull() {  # step_pull <label> <repo-dir>
  local label="$1" dir="$2" out
  if out=$(git -C "$dir" pull --ff-only 2>&1); then
    printf '[OK]   %-32s %s\n' "$label" "$(printf '%s' "$out" | tail -1)"
    OK_COUNT=$((OK_COUNT + 1))
  else
    printf '[FAIL] %-32s\n%s\n' "$label" "$out"
    FAILED+=("$label")
  fi
}

echo "===== SESSION SYNC $(date -u +%FT%TZ) ====="

echo "--- [1/4] pull ComfyUI fork"
COMFY_OLD_HEAD=$(git -C "$COMFY" rev-parse HEAD 2>/dev/null || echo "")
step_pull "ComfyUI fork" "$COMFY"
COMFY_NEW_HEAD=$(git -C "$COMFY" rev-parse HEAD 2>/dev/null || echo "")

# ComfyUI's own core deps: nothing else installs these on an existing volume
# (the manifest installer covers custom-node deps only; bootstrap runs once on
# a fresh volume). Install ONLY when this pull actually changed
# requirements.txt — zero cost on a normal spin-up.
if [ -n "$COMFY_OLD_HEAD" ] && [ "$COMFY_OLD_HEAD" != "$COMFY_NEW_HEAD" ] \
    && git -C "$COMFY" diff --name-only "$COMFY_OLD_HEAD..$COMFY_NEW_HEAD" | grep -qx "requirements.txt"; then
  echo "       requirements.txt changed in this pull -> installing into comfy venv"
  if "$COMFY/.venv/bin/python" -m pip install -r "$COMFY/requirements.txt"; then
    echo "[OK]   ComfyUI core requirements"
    OK_COUNT=$((OK_COUNT + 1))
  else
    echo "[FAIL] ComfyUI core requirements"
    FAILED+=("ComfyUI core requirements")
  fi
fi

echo "--- [2/4] pull every custom node + tool repo"
for d in "$COMFY"/custom_nodes/*/ "$COMFY"/tools/*/; do
  [ -d "$d/.git" ] || continue
  step_pull "$(basename "$d")" "$d"
done

echo "--- [3/4] manifest installer (self-skipping via its state stamp)"
if (cd "$COMFY" && "$COMFY/.venv/bin/python" scripts/install_custom_nodes.py); then
  echo "[OK]   manifest installer"
  OK_COUNT=$((OK_COUNT + 1))
else
  echo "[FAIL] manifest installer"
  FAILED+=("manifest installer")
fi

echo "--- [4/4] environment report"
"$COMFY/.venv/bin/python" -c "import torch; print('[OK]   comfy venv: torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
  || { echo "[FAIL] comfy venv torch import"; FAILED+=("comfy venv"); }
"$AITK/.venv/bin/python" -c "import torch; print('[OK]   aitk venv:  torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
  || { echo "[FAIL] aitk venv torch import"; FAILED+=("aitk venv"); }
echo "       nodes: $(ls -d "$COMFY"/custom_nodes/*/ 2>/dev/null | wc -l) dirs | tools: $(ls -d "$COMFY"/tools/*/ 2>/dev/null | wc -l) dirs | model files: $(find /workspace/comfyui_models -type f 2>/dev/null | wc -l)"
echo "       volume usage: $(du -sh /workspace 2>/dev/null | cut -f1) used (probe details: python $COMFY/scripts/runpod/model_sync.py --probe)"

echo "========================================"
if [ "${#FAILED[@]}" -eq 0 ]; then
  echo "SYNC RESULT: ALL OK ($OK_COUNT steps)"
else
  echo "SYNC RESULT: ${#FAILED[@]} FAILED -> ${FAILED[*]}"
  echo "!! fix the failed steps before starting work"
fi

source "$COMFY/.venv/bin/activate"
echo "ready. ComfyUI:  python $COMFY/main.py --listen 0.0.0.0 --port 8188"
echo "training:        source /workspace/ai-toolkit/.venv/bin/activate"
