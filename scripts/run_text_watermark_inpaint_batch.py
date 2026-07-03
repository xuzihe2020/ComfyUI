#!/usr/bin/env python3
"""Batch text/watermark removal through a running ComfyUI server.

The saved UI workflow is the source of truth for node settings. This script
converts that UI workflow JSON to ComfyUI's API prompt format at runtime, then
patches only per-run values such as input image, output prefix, and optional
fixed-region coordinates.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW = Path("user/default/workflows/auto_text_watermark_fix_flux1_fill.json")
SEED_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
SKIP_WIDGET_INPUT_TYPES = {"IMAGEUPLOAD"}
SKIP_WIDGET_INPUT_NAMES = {"control_after_generate"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)


def copy_to_comfy_input(path: Path, batch_id: str, index: int) -> str:
    input_root = repo_root() / "input" / "text_watermark_batch" / batch_id
    input_root.mkdir(parents=True, exist_ok=True)
    target = input_root / f"{index:05d}_{path.name}"
    shutil.copy2(path, target)
    return target.relative_to(repo_root() / "input").as_posix()


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


def load_ui_workflow(path: Path) -> dict[str, Any]:
    return json.loads(resolve_repo_path(path).read_text())


def widget_value_stream(node: dict[str, Any], widget_input_count: int) -> list[Any]:
    values = list(node.get("widgets_values") or [])

    # KSampler stores the frontend seed-control selector in widgets_values, but
    # that is not a real API input. If it is not represented in node inputs,
    # remove it before assigning values by input order.
    if (
        node.get("type") == "KSampler"
        and len(values) == widget_input_count + 1
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


def convert_ui_workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    links = {
        link[0]: [str(link[1]), link[2]]
        for link in workflow.get("links", [])
        if len(link) >= 6
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
            "class_type": ui_node["type"],
            "inputs": inputs,
            "_meta": {"title": ui_node.get("title") or ui_node["type"]},
        }

    return prompt


def find_nodes(
    prompt: dict[str, Any],
    *,
    class_type: str | None = None,
    title_contains: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    matches = []
    for node_id, node in prompt.items():
        if class_type is not None and node.get("class_type") != class_type:
            continue
        title = str(node.get("_meta", {}).get("title", ""))
        if title_contains is not None and title_contains.lower() not in title.lower():
            continue
        matches.append((node_id, node))
    return matches


def first_node_id(prompt: dict[str, Any], class_type: str) -> str:
    matches = find_nodes(prompt, class_type=class_type)
    if not matches:
        raise ValueError(f"Workflow does not contain a {class_type} node.")
    return matches[0][0]


def remove_nodes_referencing(prompt: dict[str, Any], source_node_id: str, output_slot: int) -> None:
    target = [source_node_id, output_slot]
    remove_ids = [
        node_id
        for node_id, node in prompt.items()
        if any(value == target for value in node.get("inputs", {}).values())
    ]
    for node_id in remove_ids:
        prompt.pop(node_id, None)


def patch_input_image(prompt: dict[str, Any], image_name: str) -> None:
    load_id = first_node_id(prompt, "LoadImage")
    prompt[load_id]["inputs"]["image"] = image_name


def patch_save_prefixes(prompt: dict[str, Any], output_prefix: str, output_stem: str) -> None:
    save_nodes = find_nodes(prompt, class_type="SaveImage")
    if not save_nodes:
        raise ValueError("Workflow does not contain a SaveImage node.")

    for _, node in save_nodes:
        title = str(node.get("_meta", {}).get("title", "")).lower()
        current_prefix = str(node.get("inputs", {}).get("filename_prefix", "")).lower()
        is_compare = "compare" in title or "comparison" in title or "compare" in current_prefix
        stem = f"compare_{output_stem}" if is_compare else output_stem
        node["inputs"]["filename_prefix"] = f"{output_prefix}/{stem}"


def patch_fixed_region(prompt: dict[str, Any], args: argparse.Namespace) -> None:
    rect_nodes = find_nodes(prompt, class_type="Mask Rect Area (Advanced)")
    if rect_nodes:
        _, rect_node = rect_nodes[0]
        for arg_name, input_name in (
            ("x", "x"),
            ("y", "y"),
            ("w", "width"),
            ("h", "height"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                rect_node["inputs"][input_name] = value
        return

    missing = [name for name in ("x", "y", "w", "h") if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "--mask-mode fixed requires --x --y --w --h unless the workflow "
            "already contains a Mask Rect Area (Advanced) node."
        )

    load_id = first_node_id(prompt, "LoadImage")
    ocr_id = first_node_id(prompt, "ApplyEasyOCR")
    combine_id = first_node_id(prompt, "Masks Combine Batch")

    # The OCR annotated-image preview becomes invalid once the OCR node is
    # replaced by GetImageSize+, so remove output nodes that depended on it.
    remove_nodes_referencing(prompt, ocr_id, 0)

    prompt[ocr_id] = {
        "class_type": "GetImageSize+",
        "inputs": {"image": [load_id, 0]},
        "_meta": {"title": "Fixed: read image size"},
    }
    prompt[combine_id] = {
        "class_type": "Mask Rect Area (Advanced)",
        "inputs": {
            "x": args.x,
            "y": args.y,
            "width": args.w,
            "height": args.h,
            "image_width": [ocr_id, 0],
            "image_height": [ocr_id, 1],
            "blur_radius": 0,
        },
        "_meta": {"title": "Fixed: rectangle mask"},
    }


def patch_prompt_text(prompt: dict[str, Any], positive: str | None, negative: str | None) -> None:
    if positive is not None:
        matches = find_nodes(prompt, class_type="CLIPTextEncode", title_contains="positive")
        if not matches:
            raise ValueError("Could not find positive CLIPTextEncode node.")
        matches[0][1]["inputs"]["text"] = positive

    if negative is not None:
        matches = find_nodes(prompt, class_type="CLIPTextEncode", title_contains="negative")
        if not matches:
            raise ValueError("Could not find negative CLIPTextEncode node.")
        matches[0][1]["inputs"]["text"] = negative


def patch_node_inputs(prompt: dict[str, Any], class_type: str, values: dict[str, Any]) -> None:
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return
    matches = find_nodes(prompt, class_type=class_type)
    if not matches:
        raise ValueError(f"Could not find {class_type} node for overrides: {values}")
    for _, node in matches:
        node["inputs"].update(values)


def apply_overrides(prompt: dict[str, Any], args: argparse.Namespace) -> None:
    patch_prompt_text(prompt, args.positive_prompt, args.negative_prompt)
    patch_node_inputs(
        prompt,
        "ApplyEasyOCR",
        {"gpu": args.easyocr_gpu, "language_name": args.language_name},
    )
    patch_node_inputs(
        prompt,
        "MaskFix+",
        {
            "erode_dilate": args.mask_grow,
            "fill_holes": args.mask_fill_holes,
            "remove_isolated_pixels": args.mask_remove_isolated,
            "smooth": args.mask_smooth,
        },
    )
    patch_node_inputs(prompt, "MaskBoundingBox+", {"padding": args.crop_padding})
    patch_node_inputs(prompt, "MaskBlur+", {"amount": args.paste_blur})
    patch_node_inputs(
        prompt,
        "UNETLoader",
        {"unet_name": args.unet_name, "weight_dtype": args.weight_dtype},
    )
    patch_node_inputs(
        prompt,
        "DualCLIPLoader",
        {"clip_name1": args.clip_name1, "clip_name2": args.clip_name2},
    )
    patch_node_inputs(prompt, "VAELoader", {"vae_name": args.vae_name})
    patch_node_inputs(prompt, "FluxGuidance", {"guidance": args.guidance})
    patch_node_inputs(
        prompt,
        "KSampler",
        {
            "seed": args.seed,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler_name": args.sampler_name,
            "scheduler": args.scheduler,
            "denoise": args.denoise,
        },
    )


def randomize_seed_if_requested(prompt: dict[str, Any], workflow: dict[str, Any]) -> None:
    for ui_node in workflow.get("nodes", []):
        if ui_node.get("type") != "KSampler":
            continue
        values = list(ui_node.get("widgets_values") or [])
        if len(values) > 1 and values[1] == "randomize":
            node = prompt.get(str(ui_node["id"]))
            if node and "seed" in node.get("inputs", {}):
                node["inputs"]["seed"] = random.randint(0, 0xFFFFFFFFFFFFFFFF)


def build_prompt_from_workflow(
    workflow: dict[str, Any],
    image_name: str,
    output_stem: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = convert_ui_workflow_to_api_prompt(workflow)
    patch_input_image(prompt, image_name)

    if args.mask_mode == "fixed":
        patch_fixed_region(prompt, args)
    elif not find_nodes(prompt, class_type="ApplyEasyOCR"):
        raise ValueError("--mask-mode ocr requires an ApplyEasyOCR node in the workflow.")

    apply_overrides(prompt, args)
    if args.seed is None:
        randomize_seed_if_requested(prompt, workflow)
    patch_save_prefixes(prompt, args.output_prefix, output_stem)
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--mask-mode", choices=("ocr", "fixed"), default="ocr")
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--h", type=int)
    parser.add_argument("--language-name")
    parser.add_argument("--easyocr-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mask-grow", type=int)
    parser.add_argument("--mask-fill-holes", type=int)
    parser.add_argument("--mask-remove-isolated", type=int)
    parser.add_argument("--mask-smooth", type=int)
    parser.add_argument("--crop-padding", type=int)
    parser.add_argument("--paste-blur", type=int)
    parser.add_argument("--positive-prompt")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--unet-name")
    parser.add_argument("--weight-dtype")
    parser.add_argument("--clip-name1")
    parser.add_argument("--clip-name2")
    parser.add_argument("--vae-name")
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sampler-name")
    parser.add_argument("--scheduler")
    parser.add_argument("--denoise", type=float)
    parser.add_argument("--output-prefix", default="text_watermark_fix/batch")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-out-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_repo_path(args.input_dir)
    workflow = load_ui_workflow(args.workflow)
    paths = image_paths(input_dir, args.recursive)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No images found in {input_dir}")

    batch_id = uuid.uuid4().hex[:10]
    client_id = uuid.uuid4().hex
    if args.prompt_out_dir:
        args.prompt_out_dir.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths, start=1):
        image_name = path.name if args.dry_run else copy_to_comfy_input(path, batch_id, index)
        output_stem = path.stem
        prompt = build_prompt_from_workflow(workflow, image_name, output_stem, args)

        if args.prompt_out_dir:
            (args.prompt_out_dir / f"{path.stem}.api.json").write_text(
                json.dumps(prompt, indent=2) + "\n"
            )
        if args.dry_run:
            print(f"[dry-run {index}/{len(paths)}] built prompt for {path.name}")
            continue

        response = post_json(args.server, "/prompt", {"prompt": prompt, "client_id": client_id})
        prompt_id = response["prompt_id"]
        print(f"[{index}/{len(paths)}] queued {path.name}: {prompt_id}")
        if not args.no_wait:
            history = wait_for_history(args.server, prompt_id, args.timeout)
            outputs = history.get("outputs", {})
            saved = sum(len(v.get("images", [])) for v in outputs.values())
            print(f"[{index}/{len(paths)}] done {path.name}: {saved} image records")


if __name__ == "__main__":
    main()
