# Repository Instructions

## ComfyUI Workflow JSON

When creating or editing ComfyUI workflow JSON files, every node `type` value must be the exact ComfyUI registered node type.

Do not invent node type names. Do not use display names. Do not use Python class names unless that class name is exactly the key registered in `NODE_CLASS_MAPPINGS`.

Before saving a workflow that uses custom nodes, verify each custom node `type` against at least one authoritative source:

- An existing workflow exported by this ComfyUI install.
- The custom node pack's `NODE_CLASS_MAPPINGS` key.
- ComfyUI's runtime node registry.

For Impact Pack specifically, remember that some Python class names differ from workflow type names. For example, the Python class `BboxDetectorForEach` is registered for workflow JSON as `BboxDetectorSEGS`.

If a workflow contains a node title or display name, do not treat it as the node type. The only valid workflow type is the actual registered `type` string.
