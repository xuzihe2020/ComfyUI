#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh — (re)launch ComfyUI in a detached tmux session, logging to file.
# Binds to 127.0.0.1 only; reach the UI via the SSH tunnel (see README).
#
# NOTE: we deliberately do NOT use --highvram. With BF16 the 64 GB transformer
# + 36 GB text encoder exceed 80 GB if both stay resident; ComfyUI's default
# vram manager offloads the encoder after text-encode, which is what we want.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/.env" ] && { set -a; source "$HERE/.env"; set +a; }

WORKSPACE="${WORKSPACE:-/workspace}"
COMFY="${COMFY:-$WORKSPACE/ComfyUI}"
VENV="${VENV:-$WORKSPACE/venv}"
PORT="${COMFY_PORT:-8188}"
LOG="$WORKSPACE/comfyui.log"

tmux kill-session -t comfy 2>/dev/null || true
tmux new-session -d -s comfy \
  "source '$VENV/bin/activate' && cd '$COMFY' && python main.py --listen 127.0.0.1 --port $PORT 2>&1 | tee '$LOG'"

echo "ComfyUI launching in tmux session 'comfy' on 127.0.0.1:$PORT"
echo "  logs:    tail -f $LOG"
echo "  attach:  tmux attach -t comfy   (detach: Ctrl-b then d)"
echo "  stop:    tmux kill-session -t comfy"
echo
echo "From your Mac, open the UI with an SSH tunnel:"
echo "  ssh -L $PORT:localhost:$PORT runpod   # then browse http://localhost:$PORT"
