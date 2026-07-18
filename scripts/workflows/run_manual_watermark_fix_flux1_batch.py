#!/usr/bin/env python3
"""Run the manual Flux 1 Fill watermark workflow over an image folder.

The ComfyUI server must already be running. Every input image uses the same
mask and prompt, while each generation receives a fresh random seed.

Example:
    python scripts/workflows/run_manual_watermark_fix_flux1_batch.py \
        "D:\\images" "D:\\masks\\watermark.png" \
        --prompt "A clean continuation of the original photo" \
        --repeats 2 --output-dir "D:\\fixed"
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW = Path(
    "user/default/workflows/utility/manual_watermark_fix_flux1.json"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SKIP_WIDGET_INPUT_TYPES = {"IMAGEUPLOAD"}
SKIP_WIDGET_INPUT_NAMES = {"control_after_generate"}
SEED_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run manual_watermark_fix_flux1.json sequentially for every image in a folder. "
            "The same mask and prompt are used for every job."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing source images.")
    parser.add_argument("mask_image", type=Path, help="Mask image shared by every source image.")
    parser.add_argument(
        "--prompt",
        "--positive-prompt",
        dest="positive_prompt",
        help="Positive prompt override. By default, use the prompt saved in the workflow.",
    )
    parser.add_argument(
        "--negative-prompt",
        help="Negative prompt override. By default, use the negative prompt saved in the workflow.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independently seeded generations per input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination folder. Defaults to the ComfyUI output directory.",
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--recursive", action="store_true", help="Include nested input folders.")
    parser.add_argument("--limit", type=int, help="Process at most this many source images.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-generation timeout in seconds.")
    parser.add_argument(
        "--keep-staged-inputs",
        action="store_true",
        help="Keep the temporary copies placed under ComfyUI/input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build every API prompt without copying files or contacting ComfyUI.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    input_dir = args.input_dir.resolve()
    mask_image = args.mask_image.resolve()
    workflow_path = resolve_repo_path(args.workflow).resolve()
    output_dir = (args.output_dir or (repo_root() / "output")).resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not mask_image.is_file():
        raise ValueError(f"Mask image does not exist: {mask_image}")
    if mask_image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported mask image type: {mask_image.suffix}")
    if not workflow_path.is_file():
        raise ValueError(f"Workflow does not exist: {workflow_path}")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    return input_dir, mask_image, workflow_path, output_dir


def find_images(input_dir: Path, mask_image: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    mask_resolved = mask_image.resolve()
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.resolve() != mask_resolved
    )


def load_and_audit_workflow(path: Path) -> dict[str, Any]:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Workflow must contain UI/canvas nodes and links arrays.")

    node_ids = [node.get("id") for node in nodes]
    duplicate_nodes = [node_id for node_id, count in Counter(node_ids).items() if count > 1]
    if duplicate_nodes:
        raise ValueError(f"Workflow has duplicate node IDs: {duplicate_nodes}")
    nodes_by_id = {node["id"]: node for node in nodes}

    malformed = [link for link in links if not isinstance(link, list) or len(link) != 6]
    if malformed:
        raise ValueError("Workflow has malformed ComfyUI links.")
    link_ids = [link[0] for link in links]
    duplicate_links = [link_id for link_id, count in Counter(link_ids).items() if count > 1]
    if duplicate_links:
        raise ValueError(f"Workflow has duplicate link IDs: {duplicate_links}")

    target_sockets: set[tuple[int, int]] = set()
    for link_id, source_id, source_slot, target_id, target_slot, link_type in links:
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            raise ValueError(f"Workflow link {link_id} has a dangling node endpoint.")
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        outputs = source.get("outputs") or []
        inputs = target.get("inputs") or []
        if not isinstance(source_slot, int) or not 0 <= source_slot < len(outputs):
            raise ValueError(f"Workflow link {link_id} has an invalid source socket.")
        if not isinstance(target_slot, int) or not 0 <= target_slot < len(inputs):
            raise ValueError(f"Workflow link {link_id} has an invalid target socket.")
        target_key = (target_id, target_slot)
        if target_key in target_sockets:
            raise ValueError(f"Workflow has conflicting links into node {target_id} input {target_slot}.")
        target_sockets.add(target_key)
        if outputs[source_slot].get("type") != link_type or inputs[target_slot].get("type") != link_type:
            raise ValueError(f"Workflow link {link_id} has a socket type mismatch.")
        if inputs[target_slot].get("link") != link_id:
            raise ValueError(f"Workflow link {link_id} is missing from its target socket.")
        if link_id not in (outputs[source_slot].get("links") or []):
            raise ValueError(f"Workflow link {link_id} is missing from its source socket.")
    return workflow


def should_skip_widget_input(input_info: dict[str, Any]) -> bool:
    return (
        input_info.get("type") in SKIP_WIDGET_INPUT_TYPES
        or input_info.get("name") in SKIP_WIDGET_INPUT_NAMES
    )


def widget_value_stream(ui_node: dict[str, Any], widget_input_count: int) -> list[Any]:
    values = list(ui_node.get("widgets_values") or [])
    if (
        ui_node.get("type") == "KSampler"
        and len(values) == widget_input_count + 1
        and len(values) > 1
        and values[1] in SEED_CONTROL_VALUES
    ):
        values.pop(1)
    return values


def convert_ui_workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    links = {link[0]: [str(link[1]), link[2]] for link in workflow["links"]}
    prompt: dict[str, Any] = {}

    for ui_node in workflow["nodes"]:
        node_id = str(ui_node["id"])
        ui_inputs = ui_node.get("inputs") or []
        widget_input_count = sum(
            1
            for input_info in ui_inputs
            if "widget" in input_info and not should_skip_widget_input(input_info)
        )
        widget_values = widget_value_stream(ui_node, widget_input_count)
        widget_index = 0
        inputs: dict[str, Any] = {}

        for input_info in ui_inputs:
            name = input_info.get("name")
            if not name:
                continue
            has_widget = "widget" in input_info and not should_skip_widget_input(input_info)
            widget_value: Any = None
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
                    raise ValueError(f"Node {node_id} references missing link {link_id}.")
                inputs[name] = links[link_id]
            elif has_widget:
                inputs[name] = widget_value

        prompt[node_id] = {
            "class_type": ui_node["type"],
            "inputs": inputs,
            "_meta": {"title": ui_node.get("title") or ui_node["type"]},
        }
    return prompt


def unique_node(prompt: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {class_type} node, found {len(matches)}.")
    return matches[0]


def linked_node_id(value: Any, description: str) -> str:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Expected a linked node for {description}, got {value!r}.")
    return str(value[0])


def audit_api_prompt(prompt: dict[str, Any]) -> None:
    for node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and value[0] not in prompt
            ):
                raise ValueError(
                    f"API prompt node {node_id} input {input_name} references missing node {value[0]}."
                )


def build_api_prompt(
    workflow: dict[str, Any],
    staged_image_name: str,
    staged_mask_name: str,
    source_image: Path,
    positive_prompt: str | None,
    negative_prompt: str | None,
    seed: int,
    save_prefix: str,
) -> tuple[dict[str, Any], str]:
    prompt = convert_ui_workflow_to_api_prompt(copy.deepcopy(workflow))

    _, load_image = unique_node(prompt, "LoadImage")
    load_image["inputs"]["image"] = staged_image_name
    load_image["inputs"]["clean_name"] = source_image.stem
    load_image["inputs"]["root_dir"] = str(source_image.parent)

    _, load_mask = unique_node(prompt, "LoadImageMask")
    load_mask["inputs"]["image"] = staged_mask_name

    _, flux_guidance = unique_node(prompt, "FluxGuidance")
    positive_id = linked_node_id(
        flux_guidance["inputs"].get("conditioning"), "FluxGuidance conditioning"
    )
    if positive_prompt is not None:
        prompt[positive_id]["inputs"]["text"] = positive_prompt

    _, inpaint_conditioning = unique_node(prompt, "InpaintModelConditioning")
    negative_id = linked_node_id(
        inpaint_conditioning["inputs"].get("negative"),
        "InpaintModelConditioning negative",
    )
    if negative_prompt is not None:
        prompt[negative_id]["inputs"]["text"] = negative_prompt

    _, ksampler = unique_node(prompt, "KSampler")
    ksampler["inputs"]["seed"] = seed

    composite_id, _ = unique_node(prompt, "ImageCompositeMasked")
    save_id, save_image = unique_node(prompt, "SaveImage")
    save_image["inputs"]["images"] = [composite_id, 0]
    save_image["inputs"]["filename_prefix"] = save_prefix

    for node_id in list(prompt):
        if prompt[node_id].get("class_type") in {"PreviewImage", "MaskPreview", "MaskPreview+"}:
            prompt.pop(node_id)
    audit_api_prompt(prompt)
    return prompt, save_id


def copy_input(path: Path, staging_dir: Path, filename: str) -> str:
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / filename
    shutil.copy2(path, target)
    return target.relative_to(repo_root() / "input").as_posix()


def post_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to ComfyUI at {base_url}: {exc.reason}") from exc


def get_json(base_url: str, endpoint: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + endpoint, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to ComfyUI at {base_url}: {exc.reason}") from exc


def wait_for_history(base_url: str, prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        history = get_json(base_url, f"/history/{urllib.parse.quote(prompt_id)}")
        if prompt_id in history:
            result = history[prompt_id]
            status = result.get("status") or {}
            if status.get("status_str") == "error" or not status.get("completed", True):
                raise RuntimeError(
                    "ComfyUI execution failed: "
                    + json.dumps(status.get("messages") or status, ensure_ascii=False)
                )
            return result
        time.sleep(1)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for prompt {prompt_id}.")


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for counter in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output filename for {path}.")


def save_result(
    history: dict[str, Any],
    save_node_id: str,
    output_dir: Path,
    relative_source: Path,
    repeat_index: int,
    seed: int,
) -> Path:
    records = (history.get("outputs", {}).get(save_node_id, {}) or {}).get("images", [])
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one final image from SaveImage node {save_node_id}, found {len(records)}."
        )
    record = records[0]
    filename = Path(str(record["filename"])).name
    subfolder = Path(str(record.get("subfolder", "")))
    output_root = (repo_root() / "output").resolve()
    staged_file = (output_root / subfolder / filename).resolve()
    if not is_within(staged_file, output_root):
        raise RuntimeError(f"ComfyUI returned an unsafe output path: {staged_file}")
    if not staged_file.is_file():
        raise RuntimeError(f"ComfyUI output file was not found locally: {staged_file}")

    destination_dir = output_dir / relative_source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (
        f"{relative_source.stem}__repeat_{repeat_index:02d}__seed_{seed}{staged_file.suffix}"
    )
    destination = unique_destination(destination)
    if staged_file == destination.resolve():
        return staged_file
    shutil.move(str(staged_file), str(destination))
    return destination


def remove_created_staging(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if not is_within(resolved_path, resolved_parent) or resolved_path == resolved_parent:
        raise RuntimeError(f"Refusing to remove unsafe staging path: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def main() -> None:
    args = parse_args()
    try:
        input_dir, mask_image, workflow_path, output_dir = validate_args(args)
        workflow = load_and_audit_workflow(workflow_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    images = find_images(input_dir, mask_image, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")

    batch_id = uuid.uuid4().hex[:12]
    input_root = (repo_root() / "input").resolve()
    input_staging = input_root / "manual_watermark_flux1_batch" / batch_id
    output_root = (repo_root() / "output").resolve()
    output_staging = output_root / "_script_staging" / "manual_watermark_flux1_batch" / batch_id
    shared_mask_name = (
        f"manual_watermark_flux1_batch/{batch_id}/shared_mask{mask_image.suffix.lower()}"
    )
    total_jobs = len(images) * args.repeats
    client_id = uuid.uuid4().hex
    completed = False

    print(f"Workflow: {workflow_path}")
    print(f"Images: {len(images)} | repeats: {args.repeats} | total jobs: {total_jobs}")
    print(f"Output: {output_dir}")

    try:
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            copy_input(mask_image, input_staging, f"shared_mask{mask_image.suffix.lower()}")

        job_index = 0
        for image_index, source_image in enumerate(images, start=1):
            relative_source = source_image.relative_to(input_dir)
            staged_image_name = (
                f"manual_watermark_flux1_batch/{batch_id}/{image_index:05d}_{source_image.name}"
            )
            if not args.dry_run:
                staged_image_name = copy_input(
                    source_image,
                    input_staging,
                    f"{image_index:05d}_{source_image.name}",
                )

            for repeat_index in range(1, args.repeats + 1):
                job_index += 1
                seed = secrets.randbelow(MAX_SEED + 1)
                save_prefix = (
                    f"_script_staging/manual_watermark_flux1_batch/{batch_id}/"
                    f"{job_index:06d}_{source_image.stem}"
                )
                prompt, save_node_id = build_api_prompt(
                    workflow=workflow,
                    staged_image_name=staged_image_name,
                    staged_mask_name=shared_mask_name,
                    source_image=source_image,
                    positive_prompt=args.positive_prompt,
                    negative_prompt=args.negative_prompt,
                    seed=seed,
                    save_prefix=save_prefix,
                )
                label = f"[{job_index}/{total_jobs}] {relative_source} repeat {repeat_index}/{args.repeats}"
                if args.dry_run:
                    print(f"{label} | dry-run prompt valid | seed={seed}")
                    continue

                response = post_json(
                    args.server,
                    "/prompt",
                    {"prompt": prompt, "client_id": client_id},
                )
                prompt_id = response.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"ComfyUI rejected the prompt: {json.dumps(response)}")
                print(f"{label} | queued {prompt_id} | seed={seed}")
                history = wait_for_history(args.server, prompt_id, args.timeout)
                destination = save_result(
                    history,
                    save_node_id,
                    output_dir,
                    relative_source,
                    repeat_index,
                    seed,
                )
                print(f"{label} | saved {destination}")
        completed = True
    finally:
        if not args.dry_run and not args.keep_staged_inputs:
            remove_created_staging(input_staging, input_root)
        if not args.dry_run and completed:
            remove_created_staging(output_staging, output_root)

    if args.dry_run:
        print("Dry run complete; no files were copied and no prompts were submitted.")
    else:
        print(f"Complete: {total_jobs} generation(s) saved under {output_dir}")


if __name__ == "__main__":
    main()
