#!/usr/bin/env python3
r"""COPY/PASTE POWERSHELL COMMANDS

DEFAULT - 4 seeds, denoise 0.8:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER"

2 SEEDS - denoise 0.8:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    2

4 SEEDS - denoise 0.7, 0.8, and 0.9:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    4 `
    --denoise "0.7,0.8,0.9"

RESUME:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    4 `
    --denoise "0.7,0.8,0.9" `
    --resume

INCLUDE SUBFOLDERS:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    4 `
    --denoise "0.8" `
    --recursive

CUSTOM OUTPUT FOLDER:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    4 `
    --denoise "0.8" `
    --output-dir "C:\PATH\TO\OUTPUT_FOLDER"

DRY RUN:
python "C:\Users\Tony Xu\workspace\aigc\comfyui\scripts\workflows\run_flux2_inpaint_lora_batch.py" `
    "C:\PATH\TO\INPUT_FOLDER" `
    4 `
    --denoise "0.7,0.8,0.9" `
    --dry-run

INPUT FILES:
C:\PATH\TO\INPUT_FOLDER\image.png
C:\PATH\TO\INPUT_FOLDER\image_mask.png
C:\PATH\TO\INPUT_FOLDER\image.txt

DEFAULT OUTPUT:
C:\Users\Tony Xu\workspace\aigc\comfyui\output\flux2_inpaint_lora_batch
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import run_manual_watermark_fix_flux1_batch as batch_common


DEFAULT_COMPILED_WORKFLOW = Path(
    "user/default/compiled_workflows/prod/image_inpaint/flux2_inpaint_with_lora.json"
)
DEFAULT_LORA_ROOT = Path(r"C:\Users\Tony Xu\workspace\comfyui_models\lora")
DEFAULT_LORA_DIR = DEFAULT_LORA_ROOT / "flux2" / "detailed_pussy"

# ONLY these LoRAs are tested. Remove or comment out entries to run a subset.
LORA_FILENAMES = (
    "pussy_0801_20260801T214823Z.safetensors",
    # "pussy_0801_20260801T214823Z_000000450.safetensors",
    # "pussy_0801_20260801T214823Z_000000525.safetensors",
    # "pussy_0801_20260801T214823Z_000000600.safetensors",
    "pussy_0801_20260801T214823Z_000000675.safetensors",
)

DEFAULT_DENOISE = "0.8"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SEED = 0xFFFFFFFFFFFFFFFF
PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DenoiseSetting:
    text: str
    value: float


@dataclass(frozen=True)
class LoraSpec:
    path: Path
    model_name: str
    label: str


@dataclass
class ImageAsset:
    source: Path
    relative_source: Path
    mask: Path | None = None
    caption_path: Path | None = None
    caption: str | None = None
    staged_image: str | None = None
    staged_mask: str | None = None
    setup_error: Exception | None = None
    setup_traceback: str = ""


@dataclass(frozen=True)
class JobSpec:
    lora: LoraSpec
    asset: ImageAsset
    seed_index: int
    seed: int
    denoise: DenoiseSetting


@dataclass
class BatchCounts:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0


class JobStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        cause: Exception,
        *,
        prompt_id: str | None = None,
    ) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.prompt_id = prompt_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compiled Flux 2 inpaint workflow for every image/mask/caption, "
            "every LoRA, N shared seeds, and every requested denoise value."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing images, masks, and TXT captions.")
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=4,
        help="Number of unique random seeds generated per source image.",
    )
    parser.add_argument(
        "--denoise",
        default=DEFAULT_DENOISE,
        help='Comma-separated denoise values, for example "0.7,0.8,0.9".',
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_COMPILED_WORKFLOW)
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--lora-root", type=Path, default=DEFAULT_LORA_ROOT)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Include nested input folders.")
    parser.add_argument("--limit", type=int, help="Process at most this many source images.")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-generation timeout in seconds.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the saved seed plan and skip outputs that already exist.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed job.")
    parser.add_argument("--keep-staged-inputs", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every prompt without staging files or contacting ComfyUI.",
    )
    return parser.parse_args()


def parse_denoise_values(value: str) -> tuple[DenoiseSetting, ...]:
    settings: list[DenoiseSetting] = []
    seen: set[Decimal] = set()
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped:
            raise ValueError("--denoise contains an empty value")
        try:
            decimal = Decimal(stripped)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid denoise value: {stripped!r}") from exc
        if not decimal.is_finite() or not Decimal("0") <= decimal <= Decimal("1"):
            raise ValueError(f"Denoise must be between 0 and 1: {stripped!r}")
        if decimal in seen:
            raise ValueError(f"Duplicate denoise value: {stripped!r}")
        seen.add(decimal)
        canonical = format(decimal.normalize(), "f")
        if "." not in canonical:
            canonical += ".0"
        settings.append(DenoiseSetting(canonical, float(decimal)))
    if not settings:
        raise ValueError("At least one denoise value is required")
    return tuple(settings)


def validate_args(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, tuple[DenoiseSetting, ...]]:
    input_dir = args.input_dir.resolve()
    workflow_path = batch_common.resolve_repo_path(args.workflow).resolve()
    lora_root = args.lora_root.resolve()
    lora_dir = args.lora_dir.resolve()
    output_dir = (
        args.output_dir
        or (batch_common.repo_root() / "output" / "flux2_inpaint_lora_batch")
    ).resolve()
    denoise_values = parse_denoise_values(args.denoise)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not workflow_path.is_file():
        raise ValueError(f"Compiled workflow does not exist: {workflow_path}")
    if not lora_root.is_dir():
        raise ValueError(f"LoRA root does not exist: {lora_root}")
    if not lora_dir.is_dir():
        raise ValueError(f"LoRA directory does not exist: {lora_dir}")
    try:
        lora_dir.relative_to(lora_root)
    except ValueError as exc:
        raise ValueError(f"LoRA directory must remain beneath {lora_root}") from exc
    if args.n < 1:
        raise ValueError("N must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.timeout < 1:
        raise ValueError("--timeout must be at least 1")
    return input_dir, workflow_path, lora_root, lora_dir, output_dir, denoise_values


def discover_source_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and path.suffix.casefold() in IMAGE_EXTENSIONS
            and not path.stem.casefold().endswith("_mask")
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _matching_sibling_files(path: Path, stem: str, suffixes: set[str]) -> list[Path]:
    return sorted(
        (
            candidate
            for candidate in path.parent.iterdir()
            if candidate.is_file()
            and candidate.stem.casefold() == stem.casefold()
            and candidate.suffix.casefold() in suffixes
        ),
        key=lambda candidate: candidate.name.casefold(),
    )


def find_mask(image_path: Path) -> Path:
    candidates = _matching_sibling_files(
        image_path, f"{image_path.stem}_mask", IMAGE_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"Mask not found for {image_path.name}; expected {image_path.stem}_mask.*")
    same_extension = [path for path in candidates if path.suffix.casefold() == image_path.suffix.casefold()]
    if len(same_extension) == 1:
        return same_extension[0]
    png = [path for path in candidates if path.suffix.casefold() == ".png"]
    if len(png) == 1:
        return png[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Multiple masks match {image_path.name}: " + ", ".join(path.name for path in candidates)
    )


def find_caption(image_path: Path) -> Path:
    candidates = _matching_sibling_files(image_path, image_path.stem, {".txt"})
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(f"Caption not found for {image_path.name}; expected {image_path.stem}.txt")
        raise ValueError(
            f"Multiple captions match {image_path.name}: "
            + ", ".join(path.name for path in candidates)
        )
    return candidates[0]


def read_caption(path: Path) -> str:
    errors: list[UnicodeError] = []
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            caption = path.read_text(encoding=encoding).strip()
            if not caption:
                raise ValueError(f"Caption is empty: {path}")
            return caption
        except UnicodeError as exc:
            errors.append(exc)
    raise ValueError(f"Could not decode caption {path}: {errors[-1]}")


def discover_assets(input_dir: Path, recursive: bool, limit: int | None) -> list[ImageAsset]:
    images = discover_source_images(input_dir, recursive)
    if limit is not None:
        images = images[:limit]
    assets: list[ImageAsset] = []
    for image in images:
        asset = ImageAsset(source=image, relative_source=image.relative_to(input_dir))
        try:
            asset.mask = find_mask(image)
            asset.caption_path = find_caption(image)
            asset.caption = read_caption(asset.caption_path)
        except Exception as exc:
            asset.setup_error = exc
            asset.setup_traceback = traceback.format_exc()
        assets.append(asset)
    return assets


def discover_loras(
    lora_dir: Path,
    lora_root: Path,
    filenames: tuple[str, ...] = LORA_FILENAMES,
) -> tuple[LoraSpec, ...]:
    if not filenames:
        raise ValueError("LORA_FILENAMES is empty; select at least one LoRA")
    normalized = [name.casefold() for name in filenames]
    if len(normalized) != len(set(normalized)):
        raise ValueError("LORA_FILENAMES contains duplicate entries")

    paths = tuple(lora_dir / filename for filename in filenames)
    invalid = [path.name for path in paths if path.suffix.casefold() != ".safetensors"]
    if invalid:
        raise ValueError("LORA_FILENAMES contains non-.safetensors entries: " + ", ".join(invalid))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Selected LoRA files do not exist: " + ", ".join(missing))

    specs: list[LoraSpec] = []
    for path in paths:
        relative = path.relative_to(lora_root)
        model_name = str(PureWindowsPath(*relative.parts))
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "lora"
        specs.append(LoraSpec(path, model_name, label))
    return tuple(specs)


def load_compiled_workflow(path: Path) -> dict[str, Any]:
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("prompt"), dict):
        raise ValueError("Compiled workflow sidecar must contain an API prompt object")
    workflow_path = sidecar.get("workflow_path")
    workflow_hash = sidecar.get("workflow_sha256")
    if not isinstance(workflow_path, str) or not workflow_path.startswith("workflows/prod/"):
        raise ValueError("Compiled workflow must reference a production UI workflow")
    source_workflow = batch_common.repo_root() / "user" / "default" / Path(*workflow_path.split("/"))
    if not source_workflow.is_file():
        raise ValueError(f"Source UI workflow is missing: {source_workflow}")
    actual_hash = hashlib.sha256(source_workflow.read_bytes()).hexdigest()
    if workflow_hash != actual_hash:
        raise ValueError(
            "Compiled workflow is stale. Open and save its source workflow in ComfyUI."
        )
    batch_common.audit_api_prompt(sidecar["prompt"])
    return sidecar


def endpoint_node_id(sidecar: dict[str, Any], endpoint_key: str) -> str:
    matches = [
        str(binding.get("node_id"))
        for binding in sidecar.get("bindings", [])
        if isinstance(binding, dict)
        and isinstance(binding.get("endpoint"), dict)
        and binding["endpoint"].get("key") == endpoint_key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one compiled binding for {endpoint_key}, found {len(matches)}")
    node_id = matches[0]
    if node_id not in sidecar["prompt"]:
        raise ValueError(f"Binding {endpoint_key} points to missing prompt node {node_id}")
    return node_id


def build_api_prompt(
    sidecar: dict[str, Any],
    *,
    staged_image: str,
    staged_mask: str,
    source_image: Path,
    caption: str,
    lora_name: str,
    seed: int,
    denoise: float,
    save_prefix: str,
) -> tuple[dict[str, Any], str]:
    prompt = copy.deepcopy(sidecar["prompt"])
    image_node = prompt[endpoint_node_id(sidecar, "input_image_01")]
    mask_node = prompt[endpoint_node_id(sidecar, "input_mask_01")]
    prompt_node = prompt[endpoint_node_id(sidecar, "prompt_01")]
    lora_node = prompt[endpoint_node_id(sidecar, "lora_01")]
    denoise_node = prompt[endpoint_node_id(sidecar, "denoise_01")]
    image_node["inputs"]["image"] = staged_image
    image_node["inputs"]["clean_name"] = source_image.stem
    image_node["inputs"]["root_dir"] = str(source_image.parent)
    mask_node["inputs"]["image"] = staged_mask
    prompt_node["inputs"]["text"] = caption
    lora_node["inputs"]["lora_name"] = lora_name
    denoise_node["inputs"]["denoise"] = denoise
    _, noise_node = batch_common.unique_node(prompt, "RandomNoise")
    noise_node["inputs"]["noise_seed"] = seed
    save_id, save_node = batch_common.unique_node(prompt, "SaveImage")
    save_node["inputs"]["filename_prefix"] = save_prefix
    for node_id in list(prompt):
        if prompt[node_id].get("class_type") in {"PreviewImage", "MaskPreview", "MaskPreview+"}:
            prompt.pop(node_id)
    batch_common.audit_api_prompt(prompt)
    return prompt, save_id


def generate_unique_seed_map(
    relative_sources: Iterable[Path],
    n: int,
    randbelow: Callable[[int], int] = secrets.randbelow,
) -> dict[str, list[int]]:
    used: set[int] = set()
    result: dict[str, list[int]] = {}
    for relative_source in relative_sources:
        seeds: list[int] = []
        while len(seeds) < n:
            seed = randbelow(MAX_SEED + 1)
            if seed in used:
                continue
            used.add(seed)
            seeds.append(seed)
        result[relative_source.as_posix()] = seeds
    return result


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_or_create_seed_plan(
    *,
    plan_path: Path,
    input_dir: Path,
    assets: list[ImageAsset],
    n: int,
    loras: tuple[LoraSpec, ...],
    denoise_values: tuple[DenoiseSetting, ...],
    resume: bool,
) -> dict[str, list[int]]:
    expected_loras = [lora.model_name for lora in loras]
    expected_denoise = [setting.text for setting in denoise_values]
    if resume:
        if not plan_path.is_file():
            raise ValueError(f"--resume requires an existing seed plan: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "input_dir": str(input_dir),
            "n": n,
            "loras": expected_loras,
            "denoise": expected_denoise,
        }
        for key, value in expected.items():
            if plan.get(key) != value:
                raise ValueError(f"Seed plan mismatch for {key}: expected {value!r}, found {plan.get(key)!r}")
        seeds = plan.get("seeds")
        if not isinstance(seeds, dict):
            raise ValueError("Seed plan is missing its seeds object")
        for asset in assets:
            values = seeds.get(asset.relative_source.as_posix())
            if not isinstance(values, list) or len(values) != n or not all(isinstance(seed, int) for seed in values):
                raise ValueError(f"Seed plan is invalid for {asset.relative_source}")
        return seeds

    seeds = generate_unique_seed_map((asset.relative_source for asset in assets), n)
    _atomic_json_write(
        plan_path,
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "n": n,
            "loras": expected_loras,
            "denoise": expected_denoise,
            "seeds": seeds,
        },
    )
    return seeds


def iter_jobs(
    loras: tuple[LoraSpec, ...],
    assets: list[ImageAsset],
    seed_map: dict[str, list[int]],
    denoise_values: tuple[DenoiseSetting, ...],
) -> Iterable[JobSpec]:
    # Cache-friendly order: load one LoRA, then exhaust its image/prompt work.
    # For each seed, all denoise values are consecutive so RandomNoise is reused.
    for lora in loras:
        for asset in assets:
            for seed_index, seed in enumerate(
                seed_map[asset.relative_source.as_posix()], start=1
            ):
                for denoise in denoise_values:
                    yield JobSpec(lora, asset, seed_index, seed, denoise)


def stage_copy_with_retry(source: Path, target: Path, attempts: int = 5) -> str:
    last_error: OSError | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            shutil.copy2(source, target)
            if target.stat().st_size != source.stat().st_size:
                raise OSError("staged file size does not match source")
            return target.relative_to(batch_common.repo_root() / "input").as_posix()
        except OSError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(0.25 * attempt, 2.0))
    raise RuntimeError(f"Could not stage {source} after {attempts} attempts") from last_error


def stage_assets(assets: list[ImageAsset], staging_dir: Path, dry_run: bool) -> None:
    for index, asset in enumerate(assets, start=1):
        if asset.setup_error is not None:
            continue
        try:
            image_name = batch_common.bounded_filename(
                asset.source.stem,
                prefix=f"{index:05d}_image_",
                suffix=asset.source.suffix,
                max_units=batch_common.filename_budget_for_directory(staging_dir),
            )
            assert asset.mask is not None
            mask_name = batch_common.bounded_filename(
                asset.mask.stem,
                prefix=f"{index:05d}_mask_",
                suffix=asset.mask.suffix,
                max_units=batch_common.filename_budget_for_directory(staging_dir),
            )
            relative_root = f"flux2_inpaint_lora_batch/{staging_dir.name}"
            if dry_run:
                asset.staged_image = f"{relative_root}/{image_name}"
                asset.staged_mask = f"{relative_root}/{mask_name}"
            else:
                asset.staged_image = stage_copy_with_retry(asset.source, staging_dir / image_name)
                asset.staged_mask = stage_copy_with_retry(asset.mask, staging_dir / mask_name)
        except Exception as exc:
            asset.setup_error = exc
            asset.setup_traceback = traceback.format_exc()


def denoise_label(setting: DenoiseSetting) -> str:
    return setting.text.replace("-", "m").replace(".", "p")


def result_path(output_dir: Path, job: JobSpec) -> Path:
    destination_dir = output_dir / job.asset.relative_source.parent
    suffix = (
        f"__lora_{job.lora.label}__seed_{job.seed_index:02d}_{job.seed}"
        f"__denoise_{denoise_label(job.denoise)}.png"
    )
    max_units = batch_common.filename_budget_for_directory(destination_dir)
    stem = batch_common.compact_filename_stem(
        job.asset.relative_source.stem, suffix=suffix, max_units=max_units
    )
    return destination_dir / f"{stem}{suffix}"


def save_result(
    history: dict[str, Any],
    save_node_id: str,
    destination: Path,
) -> Path:
    records = (history.get("outputs", {}).get(save_node_id, {}) or {}).get("images", [])
    if len(records) != 1:
        raise RuntimeError(f"Expected one image from SaveImage node {save_node_id}, found {len(records)}")
    record = records[0]
    output_root = (batch_common.repo_root() / "output").resolve()
    staged = (
        output_root
        / Path(str(record.get("subfolder", "")))
        / Path(str(record["filename"])).name
    ).resolve()
    if not batch_common.is_within(staged, output_root) or not staged.is_file():
        raise RuntimeError(f"ComfyUI returned an invalid output path: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_destination = batch_common.unique_destination(destination)
    batch_common.copy_output_with_retry(staged, final_destination)
    return final_destination


def prompt_is_known(server: str, prompt_id: str) -> bool:
    try:
        history = batch_common.get_json(server, f"/history/{prompt_id}")
        if prompt_id in history:
            return True
        queue = batch_common.get_json(server, "/queue")
        for key in ("queue_running", "queue_pending"):
            for item in queue.get(key, []):
                if isinstance(item, list) and len(item) > 1 and str(item[1]) == prompt_id:
                    return True
    except Exception:
        return False
    return False


def submit_prompt_with_retry(
    server: str,
    prompt: dict[str, Any],
    client_id: str,
    prompt_id: str,
    attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = batch_common.post_json(
                server,
                "/prompt",
                {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id},
            )
            accepted = response.get("prompt_id")
            if accepted != prompt_id:
                raise RuntimeError(f"ComfyUI returned unexpected prompt ID {accepted!r}")
            return prompt_id
        except Exception as exc:
            last_error = exc
            if prompt_is_known(server, prompt_id):
                return prompt_id
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise RuntimeError(f"Prompt submission failed after {attempts} attempts") from last_error


def wait_for_history_resilient(server: str, prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    transient_failures = 0
    while time.monotonic() < deadline:
        try:
            history = batch_common.get_json(server, f"/history/{prompt_id}")
            transient_failures = 0
            if prompt_id in history:
                result = history[prompt_id]
                status = result.get("status") or {}
                if status.get("status_str") == "error" or not status.get("completed", True):
                    raise RuntimeError(
                        "ComfyUI execution failed: "
                        + json.dumps(status.get("messages") or status, ensure_ascii=False)
                    )
                return result
        except RuntimeError as exc:
            if str(exc).startswith("ComfyUI execution failed:"):
                raise
            transient_failures += 1
            if transient_failures > 5:
                raise RuntimeError("History polling failed repeatedly") from exc
        time.sleep(min(1 + transient_failures, 5))
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for prompt {prompt_id}")


def validate_server(server: str, sidecar: dict[str, Any], loras: tuple[LoraSpec, ...]) -> None:
    object_info = batch_common.get_json(server, "/object_info")
    required_types = {
        node.get("class_type")
        for node in sidecar["prompt"].values()
        if isinstance(node.get("class_type"), str)
    }
    missing_types = sorted(required_types - object_info.keys())
    if missing_types:
        raise ValueError("ComfyUI is missing node types: " + ", ".join(missing_types))
    lora_info = object_info.get("LoraLoaderModelOnly", {})
    options = (
        lora_info.get("input", {})
        .get("required", {})
        .get("lora_name", [[]])[0]
    )
    normalized_options = {str(value).replace("/", "\\").casefold() for value in options}
    missing_loras = [
        lora.model_name
        for lora in loras
        if lora.model_name.replace("/", "\\").casefold() not in normalized_options
    ]
    if missing_loras:
        raise ValueError("ComfyUI has not registered LoRAs: " + ", ".join(missing_loras))


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        batch_common.log(f"Warning: could not write run manifest {path}: {exc}")


def process_jobs(
    jobs: Iterable[JobSpec],
    handler: Callable[[int, JobSpec], Literal["succeeded", "skipped"]],
    on_error: Callable[[int, JobSpec, Exception], None],
    *,
    fail_fast: bool,
    counts: BatchCounts | None = None,
) -> BatchCounts:
    counts = counts or BatchCounts()
    for job_index, job in enumerate(jobs, start=1):
        try:
            status = handler(job_index, job)
            if status == "succeeded":
                counts.succeeded += 1
            else:
                counts.skipped += 1
        except Exception as exc:
            counts.failed += 1
            on_error(job_index, job, exc)
            if fail_fast:
                raise
    return counts


def main() -> None:
    batch_common.configure_console_encoding()
    args = parse_args()
    try:
        (
            input_dir,
            workflow_path,
            lora_root,
            lora_dir,
            output_dir,
            denoise_values,
        ) = validate_args(args)
        sidecar = load_compiled_workflow(workflow_path)
        loras = discover_loras(lora_dir, lora_root)
        assets = discover_assets(input_dir, args.recursive, args.limit)
        if not assets:
            raise ValueError(f"No source images found in {input_dir}")
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = output_dir / "flux2_inpaint_lora_seed_plan.json"
        if args.dry_run and not args.resume:
            seed_map = generate_unique_seed_map(
                (asset.relative_source for asset in assets), args.n
            )
        else:
            seed_map = load_or_create_seed_plan(
                plan_path=plan_path,
                input_dir=input_dir,
                assets=assets,
                n=args.n,
                loras=loras,
                denoise_values=denoise_values,
                resume=args.resume,
            )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    batch_id = uuid.uuid4().hex[:12]
    input_root = (batch_common.repo_root() / "input").resolve()
    input_staging = input_root / "flux2_inpaint_lora_batch" / batch_id
    output_root = (batch_common.repo_root() / "output").resolve()
    output_staging = output_root / "_script_staging" / "flux2_inpaint_lora_batch" / batch_id
    manifest_path = output_dir / f"flux2_inpaint_lora_run_{batch_id}.jsonl"
    client_id = uuid.uuid4().hex
    total_jobs = len(assets) * len(loras) * args.n * len(denoise_values)

    batch_common.log(f"Compiled workflow: {workflow_path}")
    batch_common.log(f"Input images: {len(assets)} | LoRAs: {len(loras)} | seeds/image: {args.n}")
    batch_common.log(
        f"Denoise: {', '.join(setting.text for setting in denoise_values)} | total jobs: {total_jobs}"
    )
    batch_common.log("Cache order: LoRA -> image/caption/mask -> seed -> denoise")
    batch_common.log(f"Seed plan: {plan_path}")
    batch_common.log(f"Output: {output_dir}")

    if not args.dry_run:
        try:
            validate_server(args.server, sidecar, loras)
        except Exception as exc:
            raise SystemExit(f"ComfyUI preflight failed: {exc}") from exc

    stage_assets(assets, input_staging, args.dry_run)
    jobs = iter_jobs(loras, assets, seed_map, denoise_values)

    def handle_job(job_index: int, job: JobSpec) -> Literal["succeeded", "skipped"]:
        started = time.monotonic()
        destination = result_path(output_dir, job)
        label = (
            f"[{job_index}/{total_jobs}] {job.asset.relative_source} | "
            f"lora={job.lora.path.name} | seed={job.seed} | denoise={job.denoise.text}"
        )
        if args.resume and destination.is_file():
            batch_common.log(f"{label} | skipped existing {destination}")
            append_manifest(
                manifest_path,
                {"status": "skipped", "job_index": job_index, "output": str(destination)},
            )
            return "skipped"
        if job.asset.setup_error is not None:
            raise JobStageError("image_setup", job.asset.setup_error)
        assert job.asset.staged_image is not None
        assert job.asset.staged_mask is not None
        assert job.asset.caption is not None
        prompt_id: str | None = None
        stage = "build_prompt"
        try:
            save_leaf = batch_common.comfy_save_prefix_leaf(
                job.asset.source.stem, job_index, output_staging
            )
            save_prefix = (
                f"_script_staging/flux2_inpaint_lora_batch/{batch_id}/{save_leaf}"
            )
            prompt, save_node_id = build_api_prompt(
                sidecar,
                staged_image=job.asset.staged_image,
                staged_mask=job.asset.staged_mask,
                source_image=job.asset.source,
                caption=job.asset.caption,
                lora_name=job.lora.model_name,
                seed=job.seed,
                denoise=job.denoise.value,
                save_prefix=save_prefix,
            )
            if args.dry_run:
                batch_common.log(f"{label} | dry-run prompt valid")
                return "succeeded"
            prompt_id = str(uuid.uuid4())
            stage = "queue"
            submit_prompt_with_retry(args.server, prompt, client_id, prompt_id)
            batch_common.log(f"{label} | queued {prompt_id}")
            stage = "wait_for_history"
            history = wait_for_history_resilient(args.server, prompt_id, args.timeout)
            stage = "save_result"
            saved = save_result(history, save_node_id, destination)
            batch_common.log(
                f"{label} | saved {saved} | elapsed={batch_common.format_duration(time.monotonic() - started)}"
            )
            append_manifest(
                manifest_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "succeeded",
                    "job_index": job_index,
                    "source": str(job.asset.source),
                    "mask": str(job.asset.mask),
                    "caption": str(job.asset.caption_path),
                    "lora": job.lora.model_name,
                    "seed_index": job.seed_index,
                    "seed": job.seed,
                    "denoise": job.denoise.text,
                    "prompt_id": prompt_id,
                    "output": str(saved),
                },
            )
            return "succeeded"
        except JobStageError:
            raise
        except Exception as exc:
            raise JobStageError(stage, exc, prompt_id=prompt_id) from exc

    def on_error(job_index: int, job: JobSpec, error: Exception) -> None:
        stage = error.stage if isinstance(error, JobStageError) else "unknown"
        cause = error.cause if isinstance(error, JobStageError) else error
        prompt_id = error.prompt_id if isinstance(error, JobStageError) else None
        batch_common.log(
            f"[{job_index}/{total_jobs}] {job.asset.relative_source} | FAILED during {stage}: "
            f"{type(cause).__name__}: {cause} | continuing"
        )
        append_manifest(
            manifest_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "job_index": job_index,
                "source": str(job.asset.source),
                "mask": str(job.asset.mask) if job.asset.mask else None,
                "caption": str(job.asset.caption_path) if job.asset.caption_path else None,
                "lora": job.lora.model_name,
                "seed_index": job.seed_index,
                "seed": job.seed,
                "denoise": job.denoise.text,
                "prompt_id": prompt_id,
                "stage": stage,
                "error_type": type(cause).__name__,
                "error": str(cause),
                "traceback": job.asset.setup_traceback or traceback.format_exc(),
            },
        )

    counts = BatchCounts()
    try:
        process_jobs(
            jobs,
            handle_job,
            on_error,
            fail_fast=args.fail_fast,
            counts=counts,
        )
    finally:
        clean = counts.failed == 0
        if not args.dry_run and clean and not args.keep_staged_inputs:
            batch_common.best_effort_remove_created_staging(input_staging, input_root)
        if not args.dry_run and clean:
            batch_common.best_effort_remove_created_staging(output_staging, output_root)
        if not args.dry_run and not clean:
            batch_common.log(f"Staging retained because jobs failed: {input_staging}")
            batch_common.log(f"Staging retained because jobs failed: {output_staging}")

    batch_common.log(
        f"Batch complete: succeeded={counts.succeeded}, skipped={counts.skipped}, "
        f"failed={counts.failed}, total={total_jobs}"
    )
    batch_common.log(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    batch_common.run_main_with_timing(main)
