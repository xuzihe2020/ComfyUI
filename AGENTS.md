# Repository Instructions

## Response Style

Start every answer with: `My lord`

## Mandatory Before Any Command -- when shell is PowerShell

when Shell is PowerShell:
- Do not use heredocs.
- Do not use long `python -c` commands or other fragile inline scripts.
- For nontrivial JSON/workflow inspection or edits, create a checked script file with `apply_patch`, run it, then delete it if temporary.
- Never use PowerShell `ConvertTo-Json` / `ConvertFrom-Json` on workflow files.
- Before saving workflow JSON, run a graph audit.
- If you are about to run a command on workflow JSON, first state which checklist item applies. If the command would violate this checklist, do not run it.

## ComfyUI Node Naming Convention

When creating or editing ComfyUI workflow JSON, always use the exact ComfyUI registered backend node name in `type` and in `properties["Node name for S&R"]` when present.

Do not invent names, use display names, or use Python class names unless they are exactly the registered backend node name.

Do not add custom node `title` values unless the user explicitly asks.

## ComfyUI Model Paths

On Windows, the user's default external model folder is `C:\Users\Tony Xu\workspace\comfyui_models`.

Use `C:\Users\Tony Xu\workspace\comfyui\extra_model_paths.yaml` to configure or link ComfyUI model directories. Do not assume models live under the repository's local `models/` folder when the external model path is configured.

## ComfyUI Workflow Validation

Before saving any ComfyUI workflow JSON edit, audit the graph structure.

Verify that the workflow has no duplicate link IDs, no dangling input/output link references, no conflicting edges, and no source/target input-output type mismatches.

Treat files under `user/default/workflows/` as ComfyUI UI/canvas workflow JSON, not backend API prompt JSON. Preserve UI/canvas metadata that ComfyUI itself saves, including widget/helper sockets such as the unlinked `LoadImage` `upload` input used by the image upload control.

For subgraphs, audit the internal links and node sockets too. Do not leave genuinely stale saved sockets from obsolete or mismatched nodes, but do not remove frontend-only UI/canvas sockets solely because they are absent from backend `INPUT_TYPES()` when ComfyUI itself emits them in saved workflows.

Do not use PowerShell `ConvertTo-Json` / `ConvertFrom-Json` to rewrite workflow JSON because it can mangle ComfyUI link arrays. Use a safe JSON editor/script and validate that top-level `links` remain normal ComfyUI array links before saving.

## Command Safety

Do not use huge fragile inline quoted scripts such as long `python -c "..."` commands. Do not retry the same broken shell quoting pattern.

For nontrivial scripts, use a checked script file or a short reliable command. If a quoting or shell syntax error happens, stop and switch approach instead of burning turns on repeated quoting experiments.

## ComfyUI Custom Nodes

Do not edit files under `custom_nodes/` as the durable fix for a workflow or dependency problem. Custom node folders are installed artifacts and may be ignored, replaced, or recloned on another station.

If a custom node needs a source-code fix, the only acceptable durable path is:

1. fork the custom node repository under the user's GitHub account or another maintained GitHub location the user approves
2. make the source-code fix in that fork
3. push the fork
4. update `custom_nodes.manifest.json` to reference the fork
5. update `scripts/install_custom_nodes.py` only when installation behavior or dependencies need to change
6. if the old custom node folder already exists under `custom_nodes/`, remove that installed folder
7. let the user run `scripts/install_custom_nodes.py` to clone/install the fork into `custom_nodes/`

Do not create repo files whose purpose is to rewrite custom-node source files after installation. Do not apply local patches into `custom_nodes/` as the fix path.

If a custom node has an installation or startup problem that does not require changing the custom node source code, preserve the fix in one of these repo-controlled places instead:

- `custom_nodes.manifest.json`
- `scripts/install_custom_nodes.py`
- a maintained fork referenced by the manifest
- a new repo-controlled wrapper/custom node outside ignored installed artifacts

Install all required custom-node dependencies before ComfyUI starts. Do not rely on a custom node running `pip install` during ComfyUI import/startup, especially for packages with native DLLs such as OpenCV.

Do not run `scripts/install_custom_nodes.py` yourself unless the user explicitly asks you to run it in the current request. When adding or changing custom-node dependencies, update the manifest/install script and tell the user to run the installer themselves.

## Skills

Reusable operational procedures live in the top-level `skills/` folder (agent-agnostic markdown, one file per skill). This section is the index — skills are NOT auto-loaded; when a task matches a trigger below, read that skill file IN FULL before acting. When adding a new skill, add its row here, or no agent will ever find it.

| Skill file | Read it before... |
|---|---|
| `skills/runpod-pod-ops.md` | ANY RunPod work: pods, network volumes, pod SSH, deploys, terminations, cloud training/generation sessions |
| `skills/runpod-comfyui-setup.md` | setting up a fresh volume, or the FIRST command on any newly spun-up pod (mandatory session sync) |
| `skills/runpod-gdrive-rclone.md` | moving images/datasets/outputs between Google Drive and pods (contains a HARD scope rule: Drive access is pinned to the `Flux Prod` folder only) |

## RunPod Operations

Before doing anything involving RunPod (pods, network volumes, pod SSH, cloud training or generation runs), read `skills/runpod-pod-ops.md` in full — it is the operational source of truth (assets, deploy checklist, bootstrap, agent SSH access, key locations).

Core rules: pods are disposable — terminate, never stop; all durable state lives on the network volume at `/workspace`; pod IP/port change every deploy, ask the user for the current Connect line; agents never receive HF/RunPod tokens in chat — secrets are exported by the user on the pod or read from the repo `.env`.
