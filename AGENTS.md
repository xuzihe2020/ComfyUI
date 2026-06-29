# Repository Instructions

## Response Style

Start every answer with: `My lord`

## ComfyUI Node Naming Convention

When creating or editing ComfyUI workflow JSON, always use the exact ComfyUI registered backend node name in `type` and in `properties["Node name for S&R"]` when present.

Do not invent names, use display names, or use Python class names unless they are exactly the registered backend node name.

Do not add custom node `title` values unless the user explicitly asks.

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

If a custom node has an installation or startup problem, preserve the fix in one of these repo-controlled places instead:

- `custom_nodes.manifest.json`
- `script/install_custom_nodes.py`
- a maintained fork referenced by the manifest
- a new repo-controlled wrapper/custom node outside ignored installed artifacts

Install all required custom-node dependencies before ComfyUI starts. Do not rely on a custom node running `pip install` during ComfyUI import/startup, especially for packages with native DLLs such as OpenCV.
