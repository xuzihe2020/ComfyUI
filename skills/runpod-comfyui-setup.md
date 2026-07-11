# Skill: RunPod ComfyUI + ai-toolkit Setup & Session Sync

The burned-in process for standing up and re-entering pods. NON-NEGOTIABLE:
every agent follows this exactly — first-time setup on a fresh volume, and the
sync sequence on EVERY pod spin-up. Never skip the sync because "nothing
changed"; new custom nodes and dependency changes land in the fork/manifest
between sessions and MUST be applied before running anything.

Companion skill for pod lifecycle/deploy/SSH: `skills/runpod-pod-ops.md`.

## Principles

- **Tony's ComfyUI fork (`xuzihe2020/ComfyUI`) is the deployment unit.** Pods
  never clone upstream `comfyanonymous/ComfyUI`. The fork carries
  `custom_nodes.manifest.json`, `scripts/install_custom_nodes.py`, and the
  workflows — cloning it IS the configuration.
- **The manifest (`custom_nodes.manifest.json`) is the source of truth for
  ComfyUI-Manager and every custom node.** Follow it verbatim — never ask
  where a repo comes from; the manifest's `repo` field decides.
- **Install split:** Tony's forked nodes (manifest `repo` under
  `github.com/xuzihe2020/`) are `git clone`d directly, first. ComfyUI-Manager
  and all other nodes are installed by `scripts/install_custom_nodes.py`
  (which drives Manager's cm-cli) — it also installs every node's
  dependencies. ComfyUI-Manager itself lives in `custom_nodes/ComfyUI-Manager`
  like any node pack.
- **`.venv` lives inside each repo** — `/workspace/ComfyUI/.venv` (the
  installer's `comfy_python()` auto-detects it) and
  `/workspace/ai-toolkit/.venv` (separate, so trainer deps never conflict with
  node deps). Both are created with `--system-site-packages` to inherit the
  template image's torch — deploy the same template (`Runpod Pytorch 2.8.0`)
  or rebuild the venvs.
- Everything lives on `/workspace` (network volume); pods are disposable.

## First-time setup (fresh volume)

Run `scripts/runpod/bootstrap_pod.sh` (scp to pod, run in tmux, follow to
`BOOTSTRAP_DONE`). Idempotent. **Ordering rule: ALL cloning happens before any
heavy pip install** — the workspace is visibly complete early, and an
interrupted bootstrap leaves resumable state. Stages, in order:

1. Clone the ComfyUI fork to `/workspace/ComfyUI` (replaces any non-fork clone).
2. Direct-clone Tony's forked nodes (manifest repos under
   `github.com/xuzihe2020/`) into `custom_nodes/`.
3. Clone `xuzihe2020/ai-toolkit`.
4. Create both in-repo `.venv`s (no requirements yet).
5. Write `extra_model_paths.yaml` pointing at `/workspace/comfyui_models/`
   (verify folder names match the synced model tree afterwards).
6. `scripts/install_custom_nodes.py` — clones ComfyUI-Manager + the remaining
   manifest nodes via cm-cli + tools, and installs ALL node dependencies
   (pip for custom nodes lives ONLY in the install script, never in the .sh).
7. `pip install -r ComfyUI/requirements.txt` — ComfyUI's own core, the one
   thing the node installer assumes already exists (slowest step, runs late).
8. `pip install -r ai-toolkit/requirements.txt` — the trainer's core.
9. Create `/workspace/{hf-cache,datasets,configs,output,scripts,rclone}` and
   `/workspace/scripts/session_start.sh`.

Models arrive separately via `scripts/runpod/model_sync.py` (S3, no pod
needed, works on Mac and Windows; NOT Hugging Face — Tony has no HF token):
`python scripts/runpod/model_sync.py -u <path>...` uploads from the local
`comfyui_models` root to the volume's `/workspace/comfyui_models/<path>`;
`-d` downloads. Credentials come from the repo `.env` (`RUNPOD_S3_*`);
requires the AWS CLI. Volume usage/stats: `--probe` — POD ONLY (runs `du`
on the mounted filesystem, answers in seconds; there is no fast remote way,
verified: S3 has no folder sizes, RunPod has no usage API).

## Every pod spin-up (the sync — ALWAYS)

```bash
source /workspace/scripts/session_start.sh
```

The script is versioned in the fork at `scripts/runpod/session_start.sh` (the
volume copy self-updates after each fork pull). Every step prints an explicit
`[OK]`/`[FAIL]` verdict and the run ends with a scoreboard
(`SYNC RESULT: ALL OK` / `N FAILED -> <steps>`) — a failed step must be fixed
before starting work. It performs, in order — if the script is missing, do
these by hand:

1. `git pull --ff-only` in `/workspace/ComfyUI` (the fork).
2. `git pull --ff-only` in EVERY repo under `custom_nodes/` and `tools/`.
3. Run `scripts/install_custom_nodes.py` — ALWAYS. The installer is
   self-skipping: it stamps every successful run
   (`custom_nodes/.install_state.json` = manifest hash + each repo's HEAD)
   and exits in seconds when nothing changed. It does real work only when a
   pull brought commits, the manifest changed, or a folder is missing —
   which is when its dependency fixes and cm-cli passes actually matter.
   `--full` bypasses the stamp for a forced re-check.
4. Environment report (torch/CUDA in both venvs, node/tool/model counts),
   then activate `/workspace/ComfyUI/.venv` and export
   `HF_HOME=/workspace/hf-cache`.

Only after the sync: start ComfyUI (`python main.py --listen 0.0.0.0 --port
8188`) or switch to the training venv (`source
/workspace/ai-toolkit/.venv/bin/activate`).

## Verification after setup or sync

- `tmux ls` / `tail /workspace/bootstrap.log` → `BOOTSTRAP_DONE` present.
- Both venvs report the template torch: `torch 2.8.0+cu128`, cuda available.
- `ls /workspace/ComfyUI/custom_nodes/` matches the manifest folders.
- ComfyUI starts without import errors from custom nodes (the installer's job
  is that startup never needs to pip install anything).
