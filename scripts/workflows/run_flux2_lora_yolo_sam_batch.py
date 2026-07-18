#!/usr/bin/env python3
"""Batch-run the Flux 2 LoRA + YOLO/SAM face-regeneration workflow.

Prompt precedence for each image:
1. A non-empty sibling TXT file with the same stem as the image.
2. The optional global --prompt value.
3. The positive prompt saved in the workflow.

Example:
    python scripts/workflows/run_flux2_lora_yolo_sam_batch.py "D:\\images" \
        --repeats 2 --output-dir "D:\\face_regen" --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import run_manual_watermark_fix_flux1_batch as batch_common


DEFAULT_WORKFLOW = Path(
    "user/default/workflows/prod/face_regen/flux2_lora_yolo_sam.json"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run flux2_lora_yolo_sam.json sequentially for every image in a folder. "
            "A same-stem TXT file overrides the positive prompt for its image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing source images.")
    parser.add_argument(
        "--prompt",
        "--positive-prompt",
        dest="positive_prompt",
        help=(
            "Global positive-prompt override. A same-stem TXT sidecar takes precedence; "
            "without either override, the workflow prompt is used."
        ),
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip source/repeat pairs that already have a generated PNG in the output directory.",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Per-generation timeout in seconds.")
    parser.add_argument(
        "--keep-staged-inputs",
        action="store_true",
        help="Keep temporary copies placed under ComfyUI/input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build every API prompt without copying files or contacting ComfyUI.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.input_dir.resolve()
    workflow_path = batch_common.resolve_repo_path(args.workflow).resolve()
    output_dir = (args.output_dir or (batch_common.repo_root() / "output")).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not workflow_path.is_file():
        raise ValueError(f"Workflow does not exist: {workflow_path}")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    return input_dir, workflow_path, output_dir


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_prompt_sidecar(image_path: Path) -> Path | None:
    exact = image_path.with_suffix(".txt")
    if exact.is_file():
        return exact
    expected_stem = image_path.stem.casefold()
    return next(
        (
            path
            for path in image_path.parent.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".txt"
            and path.stem.casefold() == expected_stem
        ),
        None,
    )


def read_prompt_sidecar(path: Path) -> str:
    errors: list[UnicodeError] = []
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeError as exc:
            errors.append(exc)
    raise ValueError(f"Could not decode prompt sidecar {path}: {errors[-1]}")


def select_prompt_override(
    image_path: Path,
    global_prompt: str | None,
) -> tuple[str | None, str]:
    sidecar = find_prompt_sidecar(image_path)
    if sidecar is not None:
        prompt = read_prompt_sidecar(sidecar)
        if prompt:
            return prompt, f"sidecar:{sidecar.name}"
    if global_prompt is not None:
        return global_prompt, "cli"
    return None, "workflow-default"


def sanitize_and_audit_workflow(path: Path) -> dict[str, Any]:
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

    live_input_links = {
        item.get("link")
        for node in nodes
        for item in node.get("inputs", [])
        if item.get("link") is not None
    }
    stale_links = [
        link
        for link in links
        if link[1] not in nodes_by_id or link[3] not in nodes_by_id
    ]
    unsafe_stale = [link[0] for link in stale_links if link[0] in live_input_links]
    if unsafe_stale:
        raise ValueError(f"Workflow has live inputs referencing dangling links: {unsafe_stale}")
    if stale_links:
        stale_ids = {link[0] for link in stale_links}
        workflow["links"] = [link for link in links if link[0] not in stale_ids]
        for node in nodes:
            for output in node.get("outputs", []):
                if output.get("links") is not None:
                    output["links"] = [
                        link_id for link_id in output["links"] if link_id not in stale_ids
                    ] or None
        print(f"Warning: ignored unreferenced stale canvas links: {sorted(stale_ids)}")

    current_links = workflow["links"]
    current_ids = {link[0] for link in current_links}
    expected_source_links: dict[tuple[int, int], list[int]] = {}
    for link_id, source_id, source_slot, _, _, _ in current_links:
        expected_source_links.setdefault((source_id, source_slot), []).append(link_id)
    repaired_outputs = []
    for node in nodes:
        for slot, output in enumerate(node.get("outputs", [])):
            expected = expected_source_links.get((node["id"], slot), [])
            saved = output.get("links") or []
            if saved != expected:
                output["links"] = expected or None
                repaired_outputs.append(f"{node['id']}:{slot}")
    if repaired_outputs:
        print(
            "Warning: reconciled stale source socket link caches in memory: "
            + ", ".join(repaired_outputs)
        )

    target_sockets: set[tuple[int, int]] = set()
    for link_id, source_id, source_slot, target_id, target_slot, link_type in current_links:
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

    for node in nodes:
        for slot, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is not None and link_id not in current_ids:
                raise ValueError(f"Node {node['id']} input {slot} references missing link {link_id}.")
        for slot, item in enumerate(node.get("outputs", [])):
            missing = [link_id for link_id in item.get("links") or [] if link_id not in current_ids]
            if missing:
                raise ValueError(f"Node {node['id']} output {slot} references missing links {missing}.")
    return workflow


def build_api_prompt(
    workflow: dict[str, Any],
    staged_image_name: str,
    source_image: Path,
    positive_prompt: str | None,
    seed: int,
    save_prefix: str,
) -> tuple[dict[str, Any], str]:
    prompt = batch_common.convert_ui_workflow_to_api_prompt(copy.deepcopy(workflow))

    _, load_image = batch_common.unique_node(prompt, "LoadImage")
    load_image["inputs"]["image"] = staged_image_name
    load_image["inputs"]["clean_name"] = source_image.stem
    load_image["inputs"]["root_dir"] = str(source_image.parent)

    _, reference_conditioning = batch_common.unique_node(
        prompt, "FluxKontextMultiReferenceLatentMethod"
    )
    positive_id = batch_common.linked_node_id(
        reference_conditioning["inputs"].get("conditioning"),
        "FluxKontextMultiReferenceLatentMethod conditioning",
    )
    if positive_prompt is not None:
        prompt[positive_id]["inputs"]["text"] = positive_prompt

    _, random_noise = batch_common.unique_node(prompt, "RandomNoise")
    random_noise["inputs"]["noise_seed"] = seed

    save_nodes = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == "SaveImage"
    ]
    final_saves = []
    for node_id, node in save_nodes:
        image_link = node.get("inputs", {}).get("images")
        if not isinstance(image_link, list) or len(image_link) != 2:
            continue
        source_node = prompt.get(str(image_link[0]))
        if source_node and source_node.get("class_type") == "VAEDecode":
            final_saves.append((node_id, node))
    if len(final_saves) != 1:
        raise ValueError(
            f"Expected one SaveImage fed directly by VAEDecode, found {len(final_saves)}."
        )
    save_id, save_image = final_saves[0]
    save_image["inputs"]["filename_prefix"] = save_prefix
    for node_id, _ in save_nodes:
        if node_id != save_id:
            prompt.pop(node_id)

    for node_id in list(prompt):
        if prompt[node_id].get("class_type") in {"PreviewImage", "MaskPreview", "MaskPreview+"}:
            prompt.pop(node_id)
    batch_common.audit_api_prompt(prompt)
    return prompt, save_id


def main() -> None:
    batch_common.configure_console_encoding()
    args = parse_args()
    try:
        input_dir, workflow_path, output_dir = validate_args(args)
        workflow = sanitize_and_audit_workflow(workflow_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    images = find_images(input_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")

    batch_id = uuid.uuid4().hex[:12]
    input_root = (batch_common.repo_root() / "input").resolve()
    input_staging = input_root / "flux2_lora_yolo_sam_batch" / batch_id
    output_root = (batch_common.repo_root() / "output").resolve()
    output_staging = output_root / "_script_staging" / "flux2_lora_yolo_sam_batch" / batch_id
    total_jobs = len(images) * args.repeats
    client_id = uuid.uuid4().hex
    succeeded_count = 0
    skipped_count = 0
    failed_count = 0
    failure_log = None if args.dry_run else output_dir / f"flux2_face_regen_failures_{batch_id}.jsonl"

    print(f"Workflow: {workflow_path}")
    print(f"Images: {len(images)} | repeats: {args.repeats} | total jobs: {total_jobs}")
    print(f"Output: {output_dir}")

    try:
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        job_index = 0
        for image_index, source_image in enumerate(images, start=1):
            relative_source = source_image.relative_to(input_dir)
            staged_image_name = (
                f"flux2_lora_yolo_sam_batch/{batch_id}/{image_index:05d}_{source_image.name}"
            )
            setup_error: Exception | None = None
            setup_traceback = ""
            prompt_override: str | None = None
            prompt_source = "workflow-default"
            try:
                prompt_override, prompt_source = select_prompt_override(
                    source_image, args.positive_prompt
                )
                if not args.dry_run:
                    staged_image_name = batch_common.copy_input(
                        source_image,
                        input_staging,
                        f"{image_index:05d}_{source_image.name}",
                    )
            except Exception as exc:
                setup_error = exc
                setup_traceback = traceback.format_exc()

            for repeat_index in range(1, args.repeats + 1):
                job_index += 1
                label = f"[{job_index}/{total_jobs}] {relative_source} repeat {repeat_index}/{args.repeats}"
                seed: int | None = None
                prompt_id: str | None = None
                stage = "resume_check"
                try:
                    if args.resume:
                        previous = batch_common.existing_result(
                            output_dir, relative_source, repeat_index
                        )
                        if previous is not None:
                            skipped_count += 1
                            print(f"{label} | skipped existing {previous}")
                            continue

                    if setup_error is not None:
                        failed_count += 1
                        print(
                            f"{label} | FAILED during image setup: "
                            f"{type(setup_error).__name__}: {setup_error} | continuing"
                        )
                        batch_common.record_failure(
                            failure_log,
                            batch_id=batch_id,
                            job_index=job_index,
                            source_image=source_image,
                            relative_source=relative_source,
                            repeat_index=repeat_index,
                            seed=None,
                            prompt_id=None,
                            stage="image_setup",
                            error=setup_error,
                            traceback_text=setup_traceback,
                        )
                        continue

                    seed = secrets.randbelow(MAX_SEED + 1)
                    save_prefix = (
                        f"_script_staging/flux2_lora_yolo_sam_batch/{batch_id}/"
                        f"{job_index:06d}_{source_image.stem}"
                    )
                    stage = "build_prompt"
                    prompt, save_node_id = build_api_prompt(
                        workflow=workflow,
                        staged_image_name=staged_image_name,
                        source_image=source_image,
                        positive_prompt=prompt_override,
                        seed=seed,
                        save_prefix=save_prefix,
                    )
                    if args.dry_run:
                        succeeded_count += 1
                        print(
                            f"{label} | dry-run prompt valid | prompt={prompt_source} | seed={seed}"
                        )
                        continue

                    stage = "queue"
                    response = batch_common.post_json(
                        args.server,
                        "/prompt",
                        {"prompt": prompt, "client_id": client_id},
                    )
                    prompt_id = response.get("prompt_id")
                    if not prompt_id:
                        raise RuntimeError(f"ComfyUI rejected the prompt: {json.dumps(response)}")
                    print(
                        f"{label} | queued {prompt_id} | prompt={prompt_source} | seed={seed}"
                    )
                    stage = "wait_for_history"
                    history = batch_common.wait_for_history(args.server, prompt_id, args.timeout)
                    stage = "save_result"
                    destination = batch_common.save_result(
                        history,
                        save_node_id,
                        output_dir,
                        relative_source,
                        repeat_index,
                        seed,
                    )
                    succeeded_count += 1
                    print(f"{label} | saved {destination}")
                except Exception as exc:
                    failed_count += 1
                    print(
                        f"{label} | FAILED during {stage}: "
                        f"{type(exc).__name__}: {exc} | continuing"
                    )
                    batch_common.record_failure(
                        failure_log,
                        batch_id=batch_id,
                        job_index=job_index,
                        source_image=source_image,
                        relative_source=relative_source,
                        repeat_index=repeat_index,
                        seed=seed,
                        prompt_id=prompt_id,
                        stage=stage,
                        error=exc,
                        traceback_text=traceback.format_exc(),
                    )
                    continue
    finally:
        clean_staging = failed_count == 0
        if not args.dry_run and not args.keep_staged_inputs and clean_staging:
            batch_common.remove_created_staging(input_staging, input_root)
        if not args.dry_run and clean_staging:
            batch_common.remove_created_staging(output_staging, output_root)
        if not args.dry_run and not clean_staging:
            print(f"Staging retained because jobs failed: {input_staging}")
            print(f"Staging retained because jobs failed: {output_staging}")

    if args.dry_run:
        print(
            f"Dry run complete: valid={succeeded_count}, skipped={skipped_count}, "
            f"failed={failed_count}; no files were copied and no prompts were submitted."
        )
    else:
        print(
            f"Batch complete: succeeded={succeeded_count}, skipped={skipped_count}, "
            f"failed={failed_count}, total={total_jobs}"
        )
        if failed_count:
            print(f"Failure log: {failure_log}")


if __name__ == "__main__":
    main()
