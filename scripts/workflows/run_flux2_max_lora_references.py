#!/usr/bin/env python3
r"""Run the Flux.2 Max LoRA reference workflow from structured prompt JSON.

ComfyUI must already be running at --server, default http://127.0.0.1:8188.
The script loads repo-local .env before reading credentials.

Quick start
-----------
Run from the ComfyUI repo root with PowerShell:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json

Recommended first run: build prompts and debug logs without calling Flux.2:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --dry-run `
     --prompt-out-dir tmp\flux2_max_api_prompts

Common run modes
----------------
Process only the first 5 jobs:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --limit 5

Generate 3 images for each selected prompt:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --repeat 3

Process the first 4 jobs and generate 2 repeats each:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --limit 4 `
     --repeat 2

Queue jobs without waiting for completion:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --no-wait

Use a non-default ComfyUI server:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --server http://127.0.0.1:8190

Size and seed examples
----------------------
Default output size is 1024 x 1536, a 2:3 portrait frame.

Override size for the whole run:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --width 1024 `
     --height 1536

Set a fixed seed for the whole run:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --seed 123456

Notes:
- Per-job "width", "height", "seed", and "prompt_upsampling" override CLI
  defaults.
- Width and height must be between 256 and 2048 and divisible by 32.
- If no seed is fixed, each request gets a fresh random seed, including repeats.

Output and logs
---------------
Generated images are saved by ComfyUI's SaveImage node under the ComfyUI output
directory. The default filename prefix is:

   flux2_max_lora_references/batch/<output_stem>_<timestamp>

Use --output-prefix to change that output subfolder/prefix:

   .\.venv\Scripts\python.exe scripts\workflows\run_flux2_max_lora_references.py `
     --input-json path\to\flux2_jobs.json `
     --output-prefix synthetic_lora/flux2_max/v1

Every request writes a debug JSON log by default:

   logs/flux2_max_lora_references/<run>_<stem>.flux2_request.json

Change or disable logs:

   --log-dir path\to\debug_logs
   --no-log

Use --prompt-out-dir when you also want the full ComfyUI API prompt JSON:

   --prompt-out-dir tmp\flux2_max_api_prompts

Auth
----
Put the Flux/Comfy API key in the repository .env file:

   FLUX_API_KEY=replace_with_your_real_key

The first-party Flux.2 Max API node authenticates through Comfy's API proxy, so
the script maps FLUX_API_KEY to the sensitive ComfyUI prompt extra_data field
"api_key_comfy_org". The aliases BFL_API_KEY and API_KEY_COMFY_ORG are also
accepted. Real environment variables take precedence over values in .env.

Input JSON shape
----------------
The input file passed to --input-json may be any one of these:

1. A single job object:

   {
     "output_stem": "ol_office_halfbody_front_0001",
     "dressing_reference_images": ["refs/outfit_front.png", "refs/outfit_detail.png"],
     "character_reference_images": [
       "refs/face_front.png",
       "refs/face_3q_left.png",
       "refs/face_3q_right.png",
       "refs/body_front.png",
       "refs/body_side.png"
     ],
     "outfit_block": "Office lady outfit: fitted navy blazer, white blouse, pencil skirt, sheer black tights, low heels. Match the dressing references closely.",
     "shot_type": "Half-body portrait",
     "camera_view": "front 3/4 view",
     "pose": "standing naturally with relaxed shoulders, one hand lightly touching the blazer lapel",
     "expression": "soft confident smile",
     "environment_block": "Modern office interior with a clean desk and softly blurred background.",
     "lighting_camera_realism": "Soft window light from camera left, realistic shadows, 85mm portrait lens look."
   }

2. A list of job objects:

   [
     { "...": "job 1" },
     { "...": "job 2" }
   ]

3. An object containing an "items", "jobs", or "prompts" list:

   {
     "items": [
       { "...": "job 1" },
       { "...": "job 2" }
     ]
   }

Required reference fields
-------------------------
Each job should usually provide both reference lists:

- "dressing_reference_images": list[str]
  Up to 2 image paths. Aliases: "dressing_references", "dressing_refs".

- "character_reference_images": list[str]
  Up to 5 image paths. Aliases: "character_references", "character_refs".

Paths may be absolute, relative to the input JSON file, or relative to the repo
root. Supported extensions: .png, .jpg, .jpeg, .webp, .bmp.

If fewer than 2 dressing or fewer than 5 character references are provided, the
unused reference nodes are removed from the submitted ComfyUI API prompt. If more
than the allowed count is provided, the script exits with an error.

Prompt fields
-------------
The script builds the final Flux.2 Max prompt from these fields:

- "task_output_goal" or "task" or "output_goal"
- "reference_priority"
- "identity_lock" or "character_identity" or "identity"
- "outfit_block" or "outfit"
- "shot_type" or "shot" or "framing"
- "camera_view" or "angle"
- "pose" or "action"
- "expression"
- "environment_block" or "environment" or "background"
- "lighting_camera_realism" or "lighting" or "photography_style"
- "anti_drift_constraints" or "consistency_requirements"
- "avoid" or "negative_constraints"

Any of these fields may also be nested under a "chunks" object. Missing fields
fall back to the built-in master-template defaults.

The final prompt also injects an "Attached reference image order" block from the
actual reference lists. Flux.2 receives one ordered image batch, so this block is
what tells the model which attached images are dressing/outfit references and
which attached images are character identity/body references.

To bypass chunk assembly entirely, provide "prompt" or "positive_prompt"; that
string is sent directly to Flux.2 Max after the same reference-order block is
prepended.

Optional per-job overrides
--------------------------
- "output_stem": file stem used in the SaveImage prefix. Aliases: "name", "id".
- "width": output width, default from --width, initially 1024.
- "height": output height, default from --height, initially 1536.
  The defaults are a 2:3 portrait frame: 1024 x 1536.
- "seed": integer seed. If omitted, a random seed is used unless --seed is set.
- "prompt_upsampling": boolean, default from --prompt-upsampling, initially false.

Batch controls
--------------
- --limit N processes only the first N jobs from the input JSON. Default: no
  limit.
- --repeat N queues each selected job N times. Default: 1. Repeats use separate
  output prefixes. If no seed is fixed, each repeat receives a fresh random
  seed; if a job seed or --seed is set, repeats are deterministic.

Debug logging
-------------
For every queued request, and for every dry-run request, the script writes a
debug JSON file under --log-dir, default:

   logs/flux2_max_lora_references

Each log contains the final prompt sent to Flux.2 Max, output size, seed, output
prefix, and the ordered reference list with role labels. API keys are never
written to these logs.

Minimal valid job
-----------------
This runs without references, useful for pipeline testing:

{
  "output_stem": "dry_run_sample",
  "outfit_block": "A fitted black blazer over a white blouse.",
  "shot_type": "Half-body portrait",
  "camera_view": "front 3/4 view",
  "pose": "standing naturally",
  "expression": "neutral relaxed expression",
  "environment_block": "Clean daylight studio backdrop."
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW = Path("user/default/workflows/prod/lora_references/flux2_max_lora_reference_image.json")
DEFAULT_OUTPUT_PREFIX = "flux2_max_lora_references/batch"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
DEFAULT_LOG_DIR = Path("logs/flux2_max_lora_references")
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FLUX_API_KEY_ENV_KEYS = ("FLUX_API_KEY", "BFL_API_KEY", "API_KEY_COMFY_ORG")
MAX_DRESSING_REFS = 2
MAX_CHARACTER_REFS = 5
MAX_TOTAL_REFS = MAX_DRESSING_REFS + MAX_CHARACTER_REFS
MAX_SEED = 0xFFFFFFFFFFFFFFFF
SEED_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
SKIP_WIDGET_INPUT_TYPES = {"IMAGEUPLOAD"}
SKIP_WIDGET_INPUT_NAMES = {"control_after_generate"}

REF_SLOT_NAMES = [
    "Dressing Reference 1",
    "Dressing Reference 2",
    "Character Reference 1",
    "Character Reference 2",
    "Character Reference 3",
    "Character Reference 4",
    "Character Reference 5",
]
BATCH_NODE_NAMES = [
    "Reference Batch 2",
    "Reference Batch 3",
    "Reference Batch 4",
    "Reference Batch 5",
    "Reference Batch 6",
    "Reference Batch 7",
]
WORKFLOW_NODE_NAME_FLUX2_MAX = "Flux2 Max API - Generate"
WORKFLOW_NODE_NAME_SAVE = "Save Output Image"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without overwriting real env vars."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def flux_api_key() -> str:
    for key in FLUX_API_KEY_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value and "replace_with" not in value.lower() and "dummy" not in value.lower():
            return value
    return ""


def build_extra_data(api_key: str) -> dict[str, Any]:
    extra_data = {"comfy_usage_source": "flux2-max-lora-reference-script"}
    if api_key:
        extra_data["api_key_comfy_org"] = api_key
    return extra_data


def validate_flux2_dimension(name: str, value: int) -> None:
    if value < 256 or value > 2048 or value % 32 != 0:
        raise ValueError(f"{name} must be between 256 and 2048 and divisible by 32; got {value}")


def node_label(node: dict[str, Any]) -> str:
    return f"id={node.get('id')} name={saved_node_name(node)!r} type={node.get('type')!r}"


def exact_title(node: dict[str, Any]) -> str | None:
    title = node.get("title")
    return title if isinstance(title, str) else None


def sr_name(node: dict[str, Any]) -> str | None:
    properties = node.get("properties") or {}
    value = properties.get("Node name for S&R")
    return value if isinstance(value, str) else None


def saved_node_name(node: dict[str, Any]) -> str | None:
    return exact_title(node) or sr_name(node)


def unique_named_node(workflow: dict[str, Any], name: str, role: str) -> dict[str, Any]:
    matches = [node for node in workflow.get("nodes") or [] if saved_node_name(node) == name]
    if len(matches) != 1:
        details = ", ".join(node_label(node) for node in matches) or "none"
        raise ValueError(
            f"Expected exactly one {role} node named {name!r}; found {len(matches)}: {details}"
        )
    return matches[0]


def audit_workflow_graph(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Workflow must contain top-level nodes and links arrays.")

    node_by_id: dict[Any, dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id in node_by_id:
            raise ValueError(f"Duplicate node id: {node_id}")
        node_by_id[node_id] = node

    link_by_id: dict[Any, list[Any]] = {}
    target_edges: dict[tuple[Any, int], Any] = {}
    for link in links:
        if not isinstance(link, list) or len(link) < 6:
            raise ValueError(f"Invalid ComfyUI link array: {link!r}")
        link_id, source_id, source_slot, target_id, target_slot, link_type = link[:6]
        if link_id in link_by_id:
            raise ValueError(f"Duplicate link id: {link_id}")
        if source_id not in node_by_id:
            raise ValueError(f"Link {link_id} has missing source node {source_id}")
        if target_id not in node_by_id:
            raise ValueError(f"Link {link_id} has missing target node {target_id}")

        source_outputs = node_by_id[source_id].get("outputs") or []
        target_inputs = node_by_id[target_id].get("inputs") or []
        if not isinstance(source_slot, int) or source_slot >= len(source_outputs):
            raise ValueError(f"Link {link_id} has invalid source slot {source_slot}")
        if not isinstance(target_slot, int) or target_slot >= len(target_inputs):
            raise ValueError(f"Link {link_id} has invalid target slot {target_slot}")

        target_key = (target_id, target_slot)
        if target_key in target_edges:
            raise ValueError(
                f"Conflicting links {target_edges[target_key]} and {link_id} "
                f"both target node {target_id} input slot {target_slot}"
            )
        target_edges[target_key] = link_id

        source_type = source_outputs[source_slot].get("type")
        target_type = target_inputs[target_slot].get("type")
        if source_type != target_type and "*" not in {source_type, target_type}:
            raise ValueError(f"Link {link_id} type mismatch: {source_type!r} -> {target_type!r}")
        link_by_id[link_id] = link

    for node in nodes:
        node_id = node.get("id")
        for slot, input_info in enumerate(node.get("inputs") or []):
            link_id = input_info.get("link")
            if link_id is None:
                continue
            link = link_by_id.get(link_id)
            if link is None:
                raise ValueError(f"{node_label(node)} input {slot} references missing link {link_id}")
            if link[3] != node_id or link[4] != slot:
                raise ValueError(f"{node_label(node)} input {slot} conflicts with top-level link {link_id}")

        for slot, output_info in enumerate(node.get("outputs") or []):
            for link_id in output_info.get("links") or []:
                link = link_by_id.get(link_id)
                if link is None:
                    raise ValueError(f"{node_label(node)} output {slot} references missing link {link_id}")
                if link[1] != node_id or link[2] != slot:
                    raise ValueError(f"{node_label(node)} output {slot} conflicts with top-level link {link_id}")


def widget_value_stream(node: dict[str, Any], widget_input_count: int) -> list[Any]:
    values = list(node.get("widgets_values") or [])
    if len(values) == widget_input_count + 1 and len(values) > 1 and values[1] in SEED_CONTROL_VALUES:
        values.pop(1)
    return values


def should_skip_widget_input(input_info: dict[str, Any]) -> bool:
    return (
        input_info.get("type") in SKIP_WIDGET_INPUT_TYPES
        or input_info.get("name") in SKIP_WIDGET_INPUT_NAMES
    )


def convert_ui_workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    links = {
        link[0]: [str(link[1]), link[2]]
        for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6
    }

    prompt: dict[str, Any] = {}
    for ui_node in workflow.get("nodes", []):
        node_id = str(ui_node["id"])
        inputs: dict[str, Any] = {}
        ui_inputs = ui_node.get("inputs") or []
        widget_inputs = [inp for inp in ui_inputs if "widget" in inp]
        widget_values = widget_value_stream(ui_node, len(widget_inputs))
        widget_index = 0

        for input_info in ui_inputs:
            name = input_info.get("name")
            if not name:
                continue

            widget_value: Any | None = None
            has_widget = "widget" in input_info
            if has_widget:
                if widget_index >= len(widget_values):
                    raise ValueError(
                        f"Node {node_id} ({ui_node.get('type')}) has fewer widget values than widget inputs."
                    )
                widget_value = widget_values[widget_index]
                widget_index += 1

            link_id = input_info.get("link")
            if link_id is not None:
                if link_id not in links:
                    raise ValueError(f"Node {node_id} ({ui_node.get('type')}) references missing link {link_id}.")
                inputs[name] = links[link_id]
            elif has_widget and not should_skip_widget_input(input_info):
                inputs[name] = widget_value

        prompt[node_id] = {
            "class_type": ui_node["type"],
            "inputs": inputs,
            "_meta": {"title": ui_node.get("title") or ui_node["type"]},
        }

    return prompt


def api_node(prompt: dict[str, Any], ui_node: dict[str, Any]) -> dict[str, Any]:
    node_id = str(ui_node["id"])
    if node_id not in prompt:
        raise ValueError(f"Converted prompt does not contain {node_label(ui_node)}")
    return prompt[node_id]


def get_field(item: dict[str, Any], *names: str, default: Any = "") -> Any:
    chunks = item.get("chunks")
    for source in (item, chunks if isinstance(chunks, dict) else {}):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def reference_order_block(refs: list[tuple[str, Path]]) -> str:
    if not refs:
        return "Attached reference image order:\nNo reference images are attached for this run."

    dressing_indices = [index for index, (kind, _) in enumerate(refs, start=1) if kind == "dressing"]
    character_indices = [index for index, (kind, _) in enumerate(refs, start=1) if kind == "character"]
    lines = ["Attached reference image order:"]
    for index, (kind, path) in enumerate(refs, start=1):
        role = "dressing/outfit reference" if kind == "dressing" else "character face/body identity reference"
        lines.append(f"- Reference image {index}: {role}. Source file: {path.name}.")

    if dressing_indices:
        values = ", ".join(f"Reference image {index}" for index in dressing_indices)
        lines.append(f"Use {values} only for outfit, garment structure, fabric, color, styling, and dressing details.")
    if character_indices:
        values = ", ".join(f"Reference image {index}" for index in character_indices)
        lines.append(f"Use {values} only for the woman's face identity, body proportions, skin tone, age impression, and overall presence.")
    if dressing_indices and character_indices:
        lines.append(
            "Do not take the face, body, age, ethnicity, or hairstyle from the dressing references. "
            "Do not let outfit details override the character identity references."
        )
    return "\n".join(lines)


def build_prompt(item: dict[str, Any], refs: list[tuple[str, Path]]) -> str:
    direct_prompt = normalize_text(get_field(item, "prompt", "positive_prompt"))
    order_block = reference_order_block(refs)
    if direct_prompt:
        return "\n\n".join([order_block, direct_prompt])

    task = normalize_text(
        get_field(
            item,
            "task_output_goal",
            "task",
            "output_goal",
            default="Create one natural, photorealistic character reference photograph for synthetic LoRA training.",
        )
    )
    reference_priority = normalize_text(
        get_field(
            item,
            "reference_priority",
            default=(
                "- Face and identity: follow the character reference images exactly.\n"
                "- Body shape and proportions: follow the character reference images exactly.\n"
                "- Outfit: follow the dressing reference images or the outfit description below.\n"
                "- If there is conflict between the written prompt and the reference images, keep the character identity from the reference images."
            ),
        )
    )
    identity = normalize_text(
        get_field(
            item,
            "identity_lock",
            "character_identity",
            "identity",
            default=(
                "She has the same face, facial proportions, eye shape, nose shape, lips, jawline, "
                "cheek structure, skin tone, and natural facial texture as in the reference images. "
                "Her hair is long dark brown, softly wavy, with loose natural curls. Keep her body "
                "proportions consistent with the reference images, including shoulder width, waist, "
                "bust, hip proportions, and overall height impression. Her appearance should remain "
                "realistic, elegant, and natural."
            ),
        )
    )
    outfit = normalize_text(get_field(item, "outfit_block", "outfit", default="Follow the dressing reference images closely."))
    shot_type = normalize_text(get_field(item, "shot_type", "shot", "framing", default="Half-body portrait"))
    camera_view = normalize_text(get_field(item, "camera_view", "angle", default="front 3/4 view"))
    pose = normalize_text(get_field(item, "pose", "action", default="standing naturally"))
    expression = normalize_text(get_field(item, "expression", default="neutral, relaxed expression"))
    environment = normalize_text(get_field(item, "environment_block", "environment", "background", default="Simple clean studio light-colored background."))
    lighting = normalize_text(
        get_field(
            item,
            "lighting_camera_realism",
            "lighting",
            "photography_style",
            default=(
                "Use natural photorealistic lighting, soft realistic shadows, lifelike skin texture, "
                "realistic fabric texture, and professional portrait photography quality. The image "
                "should look like a real candid studio, editorial, or lifestyle photograph, not a "
                "painting, not anime, not CGI, and not an overly polished fashion render."
            ),
        )
    )
    anti_drift = normalize_text(
        get_field(
            item,
            "anti_drift_constraints",
            "consistency_requirements",
            default=(
                "Keep the same character identity, same facial structure, same hair color and hairstyle, "
                "same body proportions, and same overall appearance from the reference images. The outfit "
                "should match the outfit description closely. Create a tasteful variation in pose, angle, "
                "expression, and setting while preserving the character."
            ),
        )
    )
    avoid = normalize_text(
        get_field(
            item,
            "avoid",
            "negative_constraints",
            default=(
                "Do not generate a different person. Do not change her face, age, hairstyle, body shape, "
                "or ethnicity. Do not create a collage, contact sheet, illustration, anime image, CGI render, "
                "overly retouched beauty image, watermark, text, logo, or extra person."
            ),
        )
    )

    return "\n\n".join(
        [
            order_block,
            task,
            "The attached reference images define the same adult female character. Use the reference images as the source of truth for her facial identity, body proportions, skin tone, hair color, hairstyle, and overall appearance. Preserve her identity with high consistency across all generated images. Do not redesign her face, do not change her age, do not change her body shape, and do not change her overall visual impression.",
            f"Reference priority:\n{reference_priority}",
            f"Character identity:\n{identity}",
            f"Outfit:\n{outfit}",
            f"Shot and composition:\n{shot_type}.\nCamera view: {camera_view}.\nPose: {pose}.\nExpression: {expression}.",
            f"Scene and environment:\n{environment}",
            f"Lighting and photography style:\n{lighting}",
            f"Consistency requirements:\n{anti_drift}",
            f"Avoid:\n{avoid}",
        ]
    )


def list_field(item: dict[str, Any], *names: str) -> list[Any]:
    value = get_field(item, *names, default=[])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected one of {names} to be a list, got {type(value).__name__}")
    return value


def resolve_image_path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Reference image path must be a non-empty string, got {value!r}")
    raw = Path(value.strip())
    candidates = [raw] if raw.is_absolute() else [base_dir / raw, repo_root() / raw, raw]
    for candidate in candidates:
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension for {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(f"Reference image not found: {value}")


def reference_paths(item: dict[str, Any], base_dir: Path) -> list[tuple[str, Path]]:
    dressing = list_field(item, "dressing_reference_images", "dressing_references", "dressing_refs")
    character = list_field(item, "character_reference_images", "character_references", "character_refs")
    if len(dressing) > MAX_DRESSING_REFS:
        raise ValueError(f"Expected at most {MAX_DRESSING_REFS} dressing references, got {len(dressing)}")
    if len(character) > MAX_CHARACTER_REFS:
        raise ValueError(f"Expected at most {MAX_CHARACTER_REFS} character references, got {len(character)}")

    refs = [("dressing", resolve_image_path(path, base_dir)) for path in dressing]
    refs.extend(("character", resolve_image_path(path, base_dir)) for path in character)
    if len(refs) > MAX_TOTAL_REFS:
        raise ValueError(f"Expected at most {MAX_TOTAL_REFS} total references, got {len(refs)}")
    return refs


def clean_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "image"


def item_stem(item: dict[str, Any], index: int) -> str:
    value = get_field(item, "output_stem", "name", "id", default=f"item_{index:05d}")
    return clean_stem(str(value))


def output_prefix(base_prefix: str, stem: str, timestamp_s: int, repeat_index: int, repeat_count: int) -> str:
    repeat_suffix = f"_r{repeat_index:02d}" if repeat_count > 1 else ""
    return f"{base_prefix.strip('/')}/{stem}{repeat_suffix}_{timestamp_s}"


def flux2_node_from_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    matches = [node for node in prompt.values() if node.get("class_type") == "Flux2MaxImageNode"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Flux2MaxImageNode in API prompt; found {len(matches)}")
    return matches[0]


def write_request_log(
    log_dir: Path,
    *,
    run_index: int,
    total_runs: int,
    item_index: int,
    repeat_index: int,
    repeat_count: int,
    stem: str,
    save_prefix: str,
    refs: list[tuple[str, Path]],
    prompt: dict[str, Any],
    dry_run: bool,
) -> Path:
    flux_node = flux2_node_from_prompt(prompt)
    inputs = flux_node.get("inputs", {})
    payload = {
        "run_index": run_index,
        "total_runs": total_runs,
        "item_index": item_index,
        "repeat_index": repeat_index,
        "repeat_count": repeat_count,
        "dry_run": dry_run,
        "output_stem": stem,
        "save_prefix": save_prefix,
        "model_node": "Flux2MaxImageNode",
        "width": inputs.get("width"),
        "height": inputs.get("height"),
        "seed": inputs.get("seed"),
        "prompt_upsampling": inputs.get("prompt_upsampling"),
        "references": [
            {
                "index": index,
                "kind": kind,
                "role": (
                    "dressing/outfit reference"
                    if kind == "dressing"
                    else "character face/body identity reference"
                ),
                "path": str(path),
                "filename": path.name,
            }
            for index, (kind, path) in enumerate(refs, start=1)
        ],
        "prompt": inputs.get("prompt", ""),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_index:05d}_{stem}.flux2_request.json"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log_path


def patch_reference_load_node(node: dict[str, Any], path: Path) -> None:
    inputs = node["inputs"]
    inputs["image"] = str(path)
    inputs["clean_name"] = path.stem
    inputs["root_dir"] = str(path.parent)


def patch_references(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    refs: list[tuple[str, Path]],
) -> None:
    active_slot_names = []
    for index, (_, path) in enumerate(refs):
        slot_name = REF_SLOT_NAMES[index]
        ui_node = unique_named_node(workflow, slot_name, "reference image")
        patch_reference_load_node(api_node(prompt, ui_node), path)
        active_slot_names.append(slot_name)

    for slot_name in REF_SLOT_NAMES:
        if slot_name not in active_slot_names:
            prompt.pop(str(unique_named_node(workflow, slot_name, "reference image")["id"]), None)

    active_node_refs = [
        [str(unique_named_node(workflow, slot_name, "reference image")["id"]), 0]
        for slot_name in active_slot_names
    ]
    flux_node = api_node(prompt, unique_named_node(workflow, WORKFLOW_NODE_NAME_FLUX2_MAX, "Flux2 Max"))

    for batch_name in BATCH_NODE_NAMES:
        prompt.pop(str(unique_named_node(workflow, batch_name, "reference batch")["id"]), None)

    if not active_node_refs:
        flux_node["inputs"].pop("images", None)
        return

    if len(active_node_refs) == 1:
        flux_node["inputs"]["images"] = active_node_refs[0]
        return

    previous_ref = active_node_refs[0]
    for idx in range(1, len(active_node_refs)):
        batch_ui_node = unique_named_node(workflow, BATCH_NODE_NAMES[idx - 1], "reference batch")
        batch_id = str(batch_ui_node["id"])
        prompt[batch_id] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": previous_ref, "image2": active_node_refs[idx]},
            "_meta": {"title": batch_ui_node.get("title") or "ImageBatch"},
        }
        previous_ref = [batch_id, 0]
    flux_node["inputs"]["images"] = previous_ref


def patch_prompt(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    item: dict[str, Any],
    refs: list[tuple[str, Path]],
    save_prefix: str,
    args: argparse.Namespace,
) -> None:
    flux_node = api_node(prompt, unique_named_node(workflow, WORKFLOW_NODE_NAME_FLUX2_MAX, "Flux2 Max"))
    save_node = api_node(prompt, unique_named_node(workflow, WORKFLOW_NODE_NAME_SAVE, "save image"))

    flux_node["inputs"]["prompt"] = build_prompt(item, refs)
    width = int(get_field(item, "width", default=args.width))
    height = int(get_field(item, "height", default=args.height))
    validate_flux2_dimension("width", width)
    validate_flux2_dimension("height", height)
    flux_node["inputs"]["width"] = width
    flux_node["inputs"]["height"] = height
    seed_value = get_field(item, "seed", default=args.seed)
    flux_node["inputs"]["seed"] = random.randint(0, MAX_SEED) if seed_value is None else int(seed_value)
    flux_node["inputs"]["prompt_upsampling"] = bool(get_field(item, "prompt_upsampling", default=args.prompt_upsampling))
    save_node["inputs"]["filename_prefix"] = save_prefix
    patch_references(workflow, prompt, refs)


def normalize_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("items", "jobs", "prompts"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            items = [data]
    else:
        raise ValueError("Input JSON must be an object, list, or object containing items/jobs/prompts.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be a JSON object.")
        normalized.append(item)
    return normalized


def post_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {exc.code}: {body}") from exc


def get_json(base_url: str, endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + endpoint, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_history(base_url: str, prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = get_json(base_url, f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output width. Default: 1024 for 2:3 portrait.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output height. Default: 1536 for 2:3 portrait.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-upsampling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, help="Process at most N jobs. Default: unlimited.")
    parser.add_argument(
        "--repeat",
        "--repeat-count",
        dest="repeat",
        type=int,
        default=1,
        help="Queue each selected prompt N times. Default: 1.",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-out-dir", type=Path)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Write per-request debug logs here. Default: logs/flux2_max_lora_references.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable per-request debug logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive when provided.")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive.")

    load_env_file(DEFAULT_ENV_FILE)
    api_key = flux_api_key()
    if not api_key and not args.dry_run:
        keys = " or ".join(FLUX_API_KEY_ENV_KEYS)
        raise SystemExit(f"Missing Flux API key. Set {keys} in {DEFAULT_ENV_FILE} or the environment.")

    input_json = resolve_repo_path(args.input_json)
    workflow_path = resolve_repo_path(args.workflow)
    workflow = load_json(workflow_path)
    audit_workflow_graph(workflow)

    items = normalize_items(load_json(input_json))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No prompt items found.")

    prompt_out_dir = resolve_repo_path(args.prompt_out_dir) if args.prompt_out_dir else None
    if prompt_out_dir:
        prompt_out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_repo_path(args.log_dir)

    client_id = uuid.uuid4().hex
    timestamp_s = int(time.time())
    total_runs = len(items) * args.repeat
    run_index = 0
    for index, item in enumerate(items, start=1):
        base_stem = item_stem(item, index)
        refs = reference_paths(item, input_json.parent)
        ref_summary = f"{sum(1 for kind, _ in refs if kind == 'dressing')} dressing, {sum(1 for kind, _ in refs if kind == 'character')} character"

        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            stem = f"{base_stem}_r{repeat_index:02d}" if args.repeat > 1 else base_stem
            prompt = convert_ui_workflow_to_api_prompt(workflow)
            save_prefix = output_prefix(args.output_prefix, base_stem, timestamp_s, repeat_index, args.repeat)
            patch_prompt(workflow, prompt, item, refs, save_prefix, args)

            if prompt_out_dir:
                (prompt_out_dir / f"{stem}.api.json").write_text(
                    json.dumps(prompt, indent=2) + "\n",
                    encoding="utf-8",
                )
            log_path = None
            if not args.no_log:
                log_path = write_request_log(
                    log_dir,
                    run_index=run_index,
                    total_runs=total_runs,
                    item_index=index,
                    repeat_index=repeat_index,
                    repeat_count=args.repeat,
                    stem=stem,
                    save_prefix=save_prefix,
                    refs=refs,
                    prompt=prompt,
                    dry_run=args.dry_run,
                )

            if args.dry_run:
                repeat_text = f" repeat {repeat_index}/{args.repeat}" if args.repeat > 1 else ""
                log_text = f"; log: {log_path}" if log_path else ""
                print(f"[dry-run {run_index}/{total_runs}] {base_stem}{repeat_text}: built prompt with {ref_summary} refs{log_text}")
                continue

            response = post_json(
                args.server,
                "/prompt",
                {
                    "prompt": prompt,
                    "client_id": client_id,
                    "extra_data": build_extra_data(api_key),
                },
            )
            prompt_id = response["prompt_id"]
            repeat_text = f" repeat {repeat_index}/{args.repeat}" if args.repeat > 1 else ""
            log_text = f"; log: {log_path}" if log_path else ""
            print(f"[{run_index}/{total_runs}] queued {base_stem}{repeat_text} ({ref_summary} refs): {prompt_id}{log_text}")
            if not args.no_wait:
                history = wait_for_history(args.server, prompt_id, args.timeout)
                outputs = history.get("outputs", {})
                saved = sum(len(value.get("images", [])) for value in outputs.values())
                print(f"[{run_index}/{total_runs}] done {base_stem}{repeat_text}: {saved} image records")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
