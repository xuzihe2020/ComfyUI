#!/usr/bin/env python3
r"""Batch-run the production automatic EasyOCR + LaMa watermark workflow.

The ComfyUI server must already be running. Each source image is staged under
ComfyUI/input, processed independently, and copied to the requested output
folder. A failed image/repeat is recorded and does not stop later jobs.

PowerShell example (run from the ComfyUI directory):
    python .\scripts\workflows\run_auto_watermark_remove_lama_batch.py `
        "D:\images" `
        --output-dir "D:\watermark_removed" `
        --repeats 2 `
        --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import run_manual_watermark_fix_flux1_batch as batch_common


DEFAULT_WORKFLOW = Path(
    "user/default/workflows/prod/preprocessing/auto_watermark_remove_lama.json"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SEED = 0xFFFFFFFFFFFFFFFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run auto_watermark_remove_lama.json sequentially for every image "
            "in a folder. Individual failures are logged and later jobs continue."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing source images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination folder. Defaults to the ComfyUI output directory.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of executions per input image.",
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
    parser.add_argument("--timeout", type=int, default=3600, help="Per-job timeout in seconds.")
    parser.add_argument(
        "--keep-staged-inputs",
        action="store_true",
        help="Keep temporary input copies even when every job succeeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every API prompt without copying files or contacting ComfyUI.",
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


def final_save_node(
    prompt: dict[str, Any],
    composite_id: str,
) -> tuple[str, dict[str, Any]]:
    matches = []
    for node_id, node in prompt.items():
        if node.get("class_type") != "SaveImage":
            continue
        image_link = node.get("inputs", {}).get("images")
        if image_link == [composite_id, 0]:
            matches.append((node_id, node))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one SaveImage fed directly by ImageCompositeMasked, "
            f"found {len(matches)}."
        )
    return matches[0]


def retain_ancestors(prompt: dict[str, Any], output_node_id: str) -> dict[str, Any]:
    keep: set[str] = set()
    pending = [output_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in keep:
            continue
        if node_id not in prompt:
            raise ValueError(f"Prompt references missing output ancestor {node_id}.")
        keep.add(node_id)
        for value in prompt[node_id].get("inputs", {}).values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
            ):
                pending.append(value[0])
    return {node_id: node for node_id, node in prompt.items() if node_id in keep}


def build_api_prompt(
    workflow: dict[str, Any],
    staged_image_name: str,
    source_image: Path,
    seed: int,
    save_prefix: str,
) -> tuple[dict[str, Any], str]:
    prompt = batch_common.convert_ui_workflow_to_api_prompt(copy.deepcopy(workflow))

    _, load_image = batch_common.unique_node(prompt, "LoadImage")
    load_image["inputs"]["image"] = staged_image_name
    if "clean_name" in load_image["inputs"]:
        load_image["inputs"]["clean_name"] = source_image.stem
    if "root_dir" in load_image["inputs"]:
        load_image["inputs"]["root_dir"] = str(source_image.parent)

    _, lama_inpaint = batch_common.unique_node(prompt, "INPAINT_InpaintWithModel")
    lama_inpaint["inputs"]["seed"] = seed

    composite_id, _ = batch_common.unique_node(prompt, "ImageCompositeMasked")
    save_id, save_image = final_save_node(prompt, composite_id)
    save_image["inputs"]["filename_prefix"] = save_prefix

    prompt = retain_ancestors(prompt, save_id)
    batch_common.audit_api_prompt(prompt)
    return prompt, save_id


def validate_server_node_types(server: str, prompt: dict[str, Any]) -> None:
    object_info = batch_common.get_json(server, "/object_info")
    required_types = {
        node["class_type"]
        for node in prompt.values()
        if isinstance(node.get("class_type"), str)
    }
    missing_types = sorted(required_types - object_info.keys())
    if missing_types:
        raise ValueError(
            "ComfyUI is missing required backend node types: " + ", ".join(missing_types)
        )


def main() -> None:
    batch_common.configure_console_encoding()
    args = parse_args()
    try:
        input_dir, workflow_path, output_dir = validate_args(args)
        workflow = batch_common.load_and_audit_workflow(workflow_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    images = find_images(input_dir, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No supported images found in {input_dir}")

    if not args.dry_run:
        try:
            preflight_prompt, _ = build_api_prompt(
                workflow=workflow,
                staged_image_name="auto_watermark_remove_lama_batch/preflight.png",
                source_image=images[0],
                seed=0,
                save_prefix="_script_staging/auto_watermark_remove_lama_batch/preflight",
            )
            validate_server_node_types(args.server, preflight_prompt)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"ComfyUI preflight failed: {exc}") from exc

    batch_id = uuid.uuid4().hex[:12]
    input_root = (batch_common.repo_root() / "input").resolve()
    input_staging = input_root / "auto_watermark_remove_lama_batch" / batch_id
    output_root = (batch_common.repo_root() / "output").resolve()
    output_staging = (
        output_root / "_script_staging" / "auto_watermark_remove_lama_batch" / batch_id
    )
    total_jobs = len(images) * args.repeats
    client_id = uuid.uuid4().hex
    succeeded_count = 0
    skipped_count = 0
    failed_count = 0
    failure_log = (
        None
        if args.dry_run
        else output_dir / f"auto_watermark_remove_lama_failures_{batch_id}.jsonl"
    )

    batch_common.log(f"Workflow: {workflow_path}")
    batch_common.log(
        f"Images: {len(images)} | repeats: {args.repeats} | total jobs: {total_jobs}"
    )
    batch_common.log(f"Output: {output_dir}")

    try:
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        job_index = 0
        for image_index, source_image in enumerate(images, start=1):
            relative_source = source_image.relative_to(input_dir)
            staged_image_name = (
                f"auto_watermark_remove_lama_batch/{batch_id}/"
                f"{image_index:05d}_{source_image.name}"
            )
            setup_error: Exception | None = None
            setup_traceback = ""
            try:
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
                label = (
                    f"[{job_index}/{total_jobs}] {relative_source} "
                    f"repeat {repeat_index}/{args.repeats}"
                )
                job_started = time.monotonic()
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
                            batch_common.log(
                                f"{label} | skipped existing {previous} | "
                                f"elapsed={batch_common.format_duration(time.monotonic() - job_started)}"
                            )
                            continue

                    if setup_error is not None:
                        raise RuntimeError(
                            f"Input staging failed: {type(setup_error).__name__}: {setup_error}"
                        ) from setup_error

                    seed = secrets.randbelow(MAX_SEED + 1)
                    save_prefix = (
                        f"_script_staging/auto_watermark_remove_lama_batch/{batch_id}/"
                        f"{job_index:06d}_{source_image.stem}"
                    )
                    stage = "build_prompt"
                    prompt, save_node_id = build_api_prompt(
                        workflow=workflow,
                        staged_image_name=staged_image_name,
                        source_image=source_image,
                        seed=seed,
                        save_prefix=save_prefix,
                    )
                    if args.dry_run:
                        succeeded_count += 1
                        batch_common.log(
                            f"{label} | dry-run prompt valid | seed={seed} | "
                            f"elapsed={batch_common.format_duration(time.monotonic() - job_started)}"
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
                        raise RuntimeError(
                            f"ComfyUI rejected the prompt: {json.dumps(response, ensure_ascii=False)}"
                        )
                    batch_common.log(
                        f"{label} | queued {prompt_id} | seed={seed} | "
                        f"elapsed={batch_common.format_duration(time.monotonic() - job_started)}"
                    )

                    stage = "wait_for_history"
                    history = batch_common.wait_for_history(
                        args.server, prompt_id, args.timeout
                    )
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
                    batch_common.log(
                        f"{label} | saved {destination} | "
                        f"elapsed={batch_common.format_duration(time.monotonic() - job_started)}"
                    )
                except Exception as exc:
                    failed_count += 1
                    traceback_text = (
                        setup_traceback
                        if setup_error is not None and stage == "resume_check"
                        else traceback.format_exc()
                    )
                    failed_stage = "stage_input" if setup_error is not None else stage
                    batch_common.log(
                        f"{label} | FAILED during {failed_stage}: "
                        f"{type(exc).__name__}: {exc} | continuing | "
                        f"elapsed={batch_common.format_duration(time.monotonic() - job_started)}"
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
                        stage=failed_stage,
                        error=exc,
                        traceback_text=traceback_text,
                    )
                    continue
    finally:
        clean_staging = failed_count == 0
        if not args.dry_run and not args.keep_staged_inputs and clean_staging:
            batch_common.best_effort_remove_created_staging(input_staging, input_root)
        if not args.dry_run and clean_staging:
            batch_common.best_effort_remove_created_staging(output_staging, output_root)
        if not args.dry_run and not clean_staging:
            batch_common.log(f"Staging retained because jobs failed: {input_staging}")
            batch_common.log(f"Staging retained because jobs failed: {output_staging}")

    if args.dry_run:
        batch_common.log(
            f"Dry run complete: valid={succeeded_count}, skipped={skipped_count}, "
            f"failed={failed_count}; no files were copied and no prompts were submitted."
        )
    else:
        batch_common.log(
            f"Batch complete: succeeded={succeeded_count}, skipped={skipped_count}, "
            f"failed={failed_count}, total={total_jobs}"
        )
        if failed_count:
            batch_common.log(f"Failure log: {failure_log}")


if __name__ == "__main__":
    batch_common.run_main_with_timing(main)
