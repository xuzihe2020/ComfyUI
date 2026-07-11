#!/usr/bin/env bash
# EVERY POD SPIN-UP — no exceptions: pull repos, pull every node, sync deps.
# Verdict-style output: every step prints [OK]/[FAIL]; run ends with a scoreboard.
# The manifest installer ALWAYS runs — it is self-skipping via its own state
# stamp (custom_nodes/.install_state.json): a no-change run exits in seconds.
# Source of truth: this file lives in the ComfyUI fork at scripts/runpod/session_start.sh;
# the copy at /workspace/scripts/session_start.sh self-updates after the fork pull.
# Process definition: skills/runpod-comfyui-setup.md
set -uo pipefail
export HF_HOME=/workspace/hf-cache
export GIT_TERMINAL_PROMPT=0
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
step_pull "ComfyUI fork" "$COMFY"
# pick up a newer version of this very script from the fork (takes effect next run)
cp -f "$COMFY/scripts/runpod/session_start.sh" /workspace/scripts/session_start.sh 2>/dev/null || true

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
