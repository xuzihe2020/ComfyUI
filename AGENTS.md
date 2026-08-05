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

## Batch and Workflow Runner Resilience

Any Python, shell, batch, or PowerShell script that processes multiple independent jobs must isolate failures per job. After batch processing begins, an ordinary job error must never escape the job loop and terminate the remaining queue.

- Catch and record failures around every job stage, including input staging, prompt construction, submission, waiting, output transfer, and cleanup.
- Log failures immediately to a durable structured manifest with enough context to retry: job index, input, repeat/variant, seed, remote job or prompt ID, failed stage, error type/message, and traceback or command output.
- Print the failure concisely and continue with the next eligible job by default. Fail-fast behavior is allowed only behind an explicit user-selected option such as `--fail-fast`.
- Treat cleanup as best-effort and non-fatal. Preserve staging artifacts when they may help recover a failed job.
- Use bounded retries for transient network, file-lock, and filesystem errors before recording the job as failed.
- Long-running batches must support safe resume or idempotent skipping so completed jobs are not needlessly repeated after interruption.
- Global preflight failures may stop before the loop only when no job can run correctly, such as an unreadable workflow, missing shared input, or invalid global configuration.
- User cancellation and termination signals may stop the run; ordinary exceptions may not.
- Do not place `set -e`, `$ErrorActionPreference = 'Stop'`, or equivalent fail-fast behavior around a job loop unless each iteration contains its own error boundary.
- Test the failure path by forcing an early job to fail and verifying that at least one later job still runs and the final summary reports succeeded, skipped, and failed counts.

## Shared library and operational infra ownership

This fork is part of a four-repository local workspace:

```
<parent>/          # locally: ~/Workspace/playground_01/aigc ; on volume: /workspace
├── aigc-shared/   # stable Python library
├── aigc-infra/    # infra, skills, pipelines, tools, and configs
├── ComfyUI/       # this repo — the deployed ComfyUI unit (nested custom_nodes/)
└── ai-toolkit/    # xuzihe2020/ai-toolkit fork — trainer + web UI
```

RunPod skills (`skills/runpod-*.md`), RunPod scripts (`scripts/runpod/`),
dataset-prep tools, `docs/`, and cross-repo orchestration all moved to
`aigc-infra` — read `../aigc-infra/AGENTS.md` and its skills index before any
RunPod, pipeline, or dataset-prep work. On RunPod, `aigc-infra`, ComfyUI, and
ai-toolkit are checkouts under `/workspace`; the stable `aigc-shared` library
is installed from an immutable pin rather than maintained as an operational
checkout.

This repo keeps only what ComfyUI itself needs: `custom_nodes.manifest.json`,
`scripts/install_custom_nodes.py`, workflows under `user/default/workflows/`,
and a few not-yet-migrated scripts.

The root `lib/` contains compatibility adapters for remaining `scripts/*`.
Do not extend it: reusable primitives belong in `aigc-shared`, while new
operational behavior belongs in `aigc-infra`. `.env.example` remains here only
for the ComfyUI-owned scripts that have not migrated or retired.
