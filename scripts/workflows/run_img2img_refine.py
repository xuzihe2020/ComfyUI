#!/usr/bin/env python3
"""Run the shared FLUX.2 img2img workflow over a folder of images.

The workflow JSON stays read-only. This script converts the saved ComfyUI
canvas workflow to an API prompt at runtime, then patches per-image values by
stable node names/titles rather than node IDs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_WORKFLOW = Path("user/default/workflows/example/img2img_flux_2.json")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SEED_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
SKIP_WIDGET_INPUT_TYPES = {"IMAGEUPLOAD"}
SKIP_WIDGET_INPUT_NAMES = {"control_after_generate"}
MAX_SEED = 0xFFFFFFFFFFFFFFFF

WORKFLOW_NODE_NAME_LOAD_IMAGE = "Load Input Image"
WORKFLOW_NODE_NAME_SAVE_OUTPUT_IMAGE = "Save Output Image"
WORKFLOW_NODE_NAME_CLIP_TEXT_ENCODE_PROMPT = "CLIP Text Encode - Prompt"
WORKFLOW_NODE_NAME_RANDOM_NOISE = "Random Noise"
WORKFLOW_NODE_NAME_WIDTH = "Width"
WORKFLOW_NODE_NAME_HEIGHT = "Height"
WORKFLOW_NODE_NAME_DENOISE = "Denoise"

INPUT_IMAGE = "image"
INPUT_FILENAME_PREFIX = "filename_prefix"
INPUT_TEXT = "text"
INPUT_NOISE_SEED = "noise_seed"
INPUT_VALUE = "value"

DEFAULT_OUTPUT_SUBFOLDER = "flux2_img2img"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_image_name(path: Path) -> str:
    return re.sub(r"_v\d+$", "", path.stem, flags=re.IGNORECASE)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).size


def error_skip(message: str) -> None:
    print(f"\033[1;31m\n!!! ERROR: {message} !!!\033[0m", file=sys.stderr, flush=True)


def load_flux_prompt(image_path: Path, clean_name: str) -> str | None:
    prompt_path = image_path.parent / "descriptions" / f"{clean_name}.json"
    if not prompt_path.exists():
        error_skip(f"{image_path.name}: missing description JSON: {prompt_path}")
        return None

    try:
        data = load_json(prompt_path)
    except Exception as exc:
        error_skip(f"{image_path.name}: could not parse {prompt_path}: {exc}")
        return None

    prompt = data.get("flux_prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        error_skip(f"{image_path.name}: missing non-empty flux_prompt in {prompt_path}")
        return None
    return prompt.strip()


def node_label(node: dict[str, Any]) -> str:
    title = node.get("title")
    node_type = node.get("type")
    node_id = node.get("id")
    return f"id={node_id} name={saved_node_name(node)!r} type={node_type!r} title={title!r}"


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
    matches = [
        node
        for node in workflow.get("nodes") or []
        if saved_node_name(node) == name
    ]
    if len(matches) != 1:
        details = ", ".join(node_label(node) for node in matches) or "none"
        raise ValueError(
            f"Expected exactly one {role} node named {name!r}; "
            f"found {len(matches)}: {details}"
        )
    return matches[0]


def expect_input(prompt_node: dict[str, Any], input_name: str, role: str) -> None:
    if input_name not in prompt_node.get("inputs", {}):
        raise ValueError(f"{role} node cannot accept input {input_name!r}: {prompt_node}")


def audit_workflow_graph(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Workflow must contain top-level nodes and links arrays.")

    node_by_id = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id in node_by_id:
            raise ValueError(f"Duplicate node id: {node_id}")
        node_by_id[node_id] = node

    link_by_id: dict[int, list[Any]] = {}
    target_edges: dict[tuple[Any, int], int] = {}
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
            raise ValueError(
                f"Link {link_id} type mismatch: {source_type!r} -> {target_type!r}"
            )

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
                raise ValueError(
                    f"{node_label(node)} input {slot} conflicts with top-level link {link_id}"
                )

        for slot, output_info in enumerate(node.get("outputs") or []):
            for link_id in output_info.get("links") or []:
                link = link_by_id.get(link_id)
                if link is None:
                    raise ValueError(f"{node_label(node)} output {slot} references missing link {link_id}")
                if link[1] != node_id or link[2] != slot:
                    raise ValueError(
                        f"{node_label(node)} output {slot} conflicts with top-level link {link_id}"
                    )


def widget_value_stream(node: dict[str, Any], widget_input_count: int) -> list[Any]:
    values = list(node.get("widgets_values") or [])

    if (
        len(values) == widget_input_count + 1
        and len(values) > 1
        and values[1] in SEED_CONTROL_VALUES
    ):
        values.pop(1)

    return values


def should_skip_widget_input(input_info: dict[str, Any]) -> bool:
    return (
        input_info.get("type") in SKIP_WIDGET_INPUT_TYPES
        or input_info.get("name") in SKIP_WIDGET_INPUT_NAMES
    )


def converted_class_type(ui_node: dict[str, Any]) -> str:
    if saved_node_name(ui_node) == WORKFLOW_NODE_NAME_DENOISE and ui_node.get("type") == "PrimitiveNode":
        return "PrimitiveFloat"
    return str(ui_node["type"])


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

        if saved_node_name(ui_node) == WORKFLOW_NODE_NAME_DENOISE and ui_node.get("type") == "PrimitiveNode":
            values = list(ui_node.get("widgets_values") or [])
            if not values:
                raise ValueError(f"Denoise primitive has no widget value: {node_label(ui_node)}")
            inputs[INPUT_VALUE] = values[0]
        else:
            for input_info in ui_inputs:
                name = input_info.get("name")
                if not name:
                    continue

                widget_value: Any | None = None
                has_widget = "widget" in input_info
                if has_widget:
                    if widget_index >= len(widget_values):
                        raise ValueError(
                            f"Node {node_id} ({ui_node.get('type')}) has fewer "
                            f"widget values than widget inputs."
                        )
                    widget_value = widget_values[widget_index]
                    widget_index += 1

                link_id = input_info.get("link")
                if link_id is not None:
                    if link_id not in links:
                        raise ValueError(
                            f"Node {node_id} ({ui_node.get('type')}) references "
                            f"missing link {link_id}."
                        )
                    inputs[name] = links[link_id]
                elif has_widget and not should_skip_widget_input(input_info):
                    inputs[name] = widget_value

        prompt[node_id] = {
            "class_type": converted_class_type(ui_node),
            "inputs": inputs,
            "_meta": {"title": ui_node.get("title") or ui_node["type"]},
        }

    return prompt


def api_node(prompt: dict[str, Any], ui_node: dict[str, Any]) -> dict[str, Any]:
    node_id = str(ui_node["id"])
    if node_id not in prompt:
        raise ValueError(f"Converted prompt does not contain {node_label(ui_node)}")
    return prompt[node_id]


def output_prefix(output_dir: Path | None, clean_name: str, timestamp_s: int) -> str:
    stem = f"{clean_name}_{timestamp_s}"
    if output_dir is None:
        return f"{DEFAULT_OUTPUT_SUBFOLDER}/{stem}"
    if output_dir.is_absolute():
        raise ValueError(
            "--output-dir must be a ComfyUI output subfolder, not an absolute path. "
            "ComfyUI's SaveImage node blocks saving outside its configured output directory."
        )
    return (output_dir / stem).as_posix()


def patch_prompt(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    image_path: Path,
    prompt_text: str,
    width: int,
    height: int,
    denoise: float,
    save_prefix: str,
    seed: int,
) -> None:
    load_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_LOAD_IMAGE, "input image")
    save_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_SAVE_OUTPUT_IMAGE, "save image")
    prompt_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_CLIP_TEXT_ENCODE_PROMPT, "prompt text")
    random_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_RANDOM_NOISE, "random seed")
    width_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_WIDTH, "width primitive")
    height_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_HEIGHT, "height primitive")
    denoise_node = unique_named_node(workflow, WORKFLOW_NODE_NAME_DENOISE, "denoise primitive")

    patches = [
        (api_node(prompt, load_node), INPUT_IMAGE, str(image_path.resolve()), WORKFLOW_NODE_NAME_LOAD_IMAGE),
        (api_node(prompt, save_node), INPUT_FILENAME_PREFIX, save_prefix, WORKFLOW_NODE_NAME_SAVE_OUTPUT_IMAGE),
        (api_node(prompt, prompt_node), INPUT_TEXT, prompt_text, WORKFLOW_NODE_NAME_CLIP_TEXT_ENCODE_PROMPT),
        (api_node(prompt, random_node), INPUT_NOISE_SEED, seed, WORKFLOW_NODE_NAME_RANDOM_NOISE),
        (api_node(prompt, width_node), INPUT_VALUE, width, WORKFLOW_NODE_NAME_WIDTH),
        (api_node(prompt, height_node), INPUT_VALUE, height, WORKFLOW_NODE_NAME_HEIGHT),
        (api_node(prompt, denoise_node), INPUT_VALUE, denoise, WORKFLOW_NODE_NAME_DENOISE),
    ]

    for node, input_name, value, role in patches:
        expect_input(node, input_name, role)
        node["inputs"][input_name] = value


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
    parser = argparse.ArgumentParser(
        description="Batch-run the shared FLUX.2 img2img workflow without editing it."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--denoise", type=float, default=0.45)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeat_count < 1:
        raise SystemExit("--repeat-count must be at least 1.")
    if not 0 <= args.denoise <= 1:
        raise SystemExit("--denoise must be between 0 and 1.")

    input_dir = resolve_repo_path(args.input_dir)
    workflow_path = resolve_repo_path(args.workflow)
    workflow = load_json(workflow_path)
    audit_workflow_graph(workflow)

    paths = image_paths(input_dir, args.recursive)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")

    if args.prompt_out_dir:
        args.prompt_out_dir.mkdir(parents=True, exist_ok=True)

    client_id = uuid.uuid4().hex
    total_jobs = 0
    skipped = 0

    for image_index, path in enumerate(paths, start=1):
        clean_name = clean_image_name(path)
        flux_prompt = load_flux_prompt(path, clean_name)
        if flux_prompt is None:
            skipped += 1
            continue

        width, height = image_size(path)

        for repeat_index in range(1, args.repeat_count + 1):
            seed = random.randint(0, MAX_SEED)
            save_prefix = output_prefix(args.output_dir, clean_name, int(time.time()))
            prompt = convert_ui_workflow_to_api_prompt(workflow)
            patch_prompt(
                workflow=workflow,
                prompt=prompt,
                image_path=path,
                prompt_text=flux_prompt,
                width=width,
                height=height,
                denoise=args.denoise,
                save_prefix=save_prefix,
                seed=seed,
            )

            total_jobs += 1
            job_label = (
                f"[{image_index}/{len(paths)} repeat {repeat_index}/{args.repeat_count}]"
            )

            if args.prompt_out_dir:
                prompt_path = args.prompt_out_dir / f"{clean_name}_r{repeat_index:02d}.api.json"
                prompt_path.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")

            if args.dry_run:
                print(
                    f"{job_label} dry-run {path.name}: "
                    f"{width}x{height}, seed={seed}, save_prefix={save_prefix}"
                )
                continue

            response = post_json(args.server, "/prompt", {"prompt": prompt, "client_id": client_id})
            prompt_id = response["prompt_id"]
            print(f"{job_label} queued {path.name}: {prompt_id} seed={seed}")

            if not args.no_wait:
                history = wait_for_history(args.server, prompt_id, args.timeout)
                outputs = history.get("outputs", {})
                saved = sum(len(value.get("images", [])) for value in outputs.values())
                print(f"{job_label} done {path.name}: {saved} image records")

    print(f"done: jobs={total_jobs}, skipped_images={skipped}")


if __name__ == "__main__":
    main()
