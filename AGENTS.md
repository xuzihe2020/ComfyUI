# Repository Instructions

## ComfyUI Workflow JSON

When creating or editing ComfyUI workflow JSON files, every node `type` value must be the exact ComfyUI registered backend node type.

Do not invent node type names. Do not use display names. Do not use Python class names unless that class name is exactly the key registered in `NODE_CLASS_MAPPINGS`.

Keep backend references as registered node types:

- `type`
- `properties["Node name for S&R"]` when that property exists

Do not add custom node `title` values to workflow JSON. Leave node `title` absent so ComfyUI displays the real UI/search node name from its registry, such as `Mask Fix` for backend type `MaskFix+`.

Do not set node `title` to backend type names such as `MaskFix+` when the UI/search name is different. Do not add descriptive custom titles such as workflow steps, comments, or plain-English labels. If a node title absolutely must be set, it must be the exact UI/search display name from ComfyUI's runtime registry or `NODE_DISPLAY_NAME_MAPPINGS`, not a guessed label and not the backend type unless the UI/search name is exactly the same.

Before saving a workflow that uses custom nodes, verify each custom node `type` against at least one authoritative source:

- An existing workflow exported by this ComfyUI install.
- The custom node pack's `NODE_CLASS_MAPPINGS` key.
- ComfyUI's runtime node registry.

For Impact Pack specifically, remember that some Python class names differ from workflow type names. For example, the Python class `BboxDetectorForEach` is registered for workflow JSON as `BboxDetectorSEGS`.

If a workflow contains a node title from an older file, do not treat it as the backend type. Remove custom node titles unless there is a verified reason to preserve an exact UI/search display name.

## ComfyUI Custom Nodes

Do not edit files under `custom_nodes/` as the durable fix for a workflow or dependency problem. Custom node folders are installed artifacts and may be ignored, replaced, or recloned on another station.

If a custom node has an installation or startup problem, preserve the fix in one of these repo-controlled places instead:

- `custom_nodes.manifest.json`
- `script/install_custom_nodes.py`
- a maintained fork referenced by the manifest
- a new repo-controlled wrapper/custom node outside ignored installed artifacts

Install all required custom-node dependencies before ComfyUI starts. Do not rely on a custom node running `pip install` during ComfyUI import/startup, especially for packages with native DLLs such as OpenCV.
