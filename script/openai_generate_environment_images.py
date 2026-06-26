#!/usr/bin/env python3
"""Batch-generate VN background images from environment briefs using OpenAI.

The script reads an environment brief folder produced by
gemini_environment_briefs.py, pairs each prompt with its source reference image,
and calls the OpenAI Images edit endpoint. Results are written into a target
directory with a JSONL manifest for downstream review and automation.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-image-2"
DEFAULT_API_URL = "https://api.openai.com/v1"
DEFAULT_SIZE = "1536x864"
DEFAULT_QUALITY = "high"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_BACKGROUND = "opaque"
TRANSIENT_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}
IMAGE_OUTPUT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def short_id(length: int = 8) -> str:
    return uuid.uuid4().hex[:length]


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Settings:
    brief_dir: Path
    output_dir: Path
    api_key: str
    model: str
    api_url: str
    organization: str
    project: str
    manifest_name: str
    run_id: str
    images_per_prompt: int
    max_images: int | None
    size: str
    quality: str
    output_format: str
    output_compression: int | None
    background: str
    input_fidelity: str | None
    user: str
    rerun_existing_prompts: bool
    dry_run: bool
    sleep_seconds: float
    timeout_seconds: float
    retries: int


@dataclass(frozen=True)
class PromptJob:
    source_image: Path
    prompt_id: str
    prompt_label: str
    prompt_text: str
    prompt_path: Path | None
    brief_json: Path | None
    environment_type: str


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Generate VN background images with OpenAI GPT Image from an environment brief folder.",
    )
    parser.add_argument(
        "brief_dir",
        type=Path,
        help="Environment brief folder containing manifest.jsonl, briefs/, and prompts/.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where generated images and generation_manifest.jsonl will be written.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI image model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"OpenAI API base URL. Default: {DEFAULT_API_URL}")
    parser.add_argument(
        "--organization",
        default=os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "",
        help="Optional OpenAI organization header. Defaults to OPENAI_ORG_ID/OPENAI_ORGANIZATION.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("OPENAI_PROJECT_ID") or "",
        help="Optional OpenAI project header. Defaults to OPENAI_PROJECT_ID.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Manifest filename inside brief_dir. Default: manifest.jsonl",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional short run ID used in output filenames. Defaults to a random ID.",
    )
    parser.add_argument(
        "--images-per-prompt",
        type=int,
        default=1,
        help="Number of images to request per prompt. Must be 1-10. Default: 1",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum total generated images to request. Omit to process all selected prompts.",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_SIZE,
        help=f"Output size. Default: {DEFAULT_SIZE}. For gpt-image-2, WIDTHxHEIGHT is supported.",
    )
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "auto"),
        default=DEFAULT_QUALITY,
        help=f"Image quality. Default: {DEFAULT_QUALITY}",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "jpeg", "webp"),
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Generated image format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--output-compression",
        type=int,
        default=None,
        help="Compression 0-100 for jpeg/webp outputs. Omit for API default.",
    )
    parser.add_argument(
        "--background",
        choices=("opaque", "auto", "transparent"),
        default=DEFAULT_BACKGROUND,
        help=f"Background mode. Default: {DEFAULT_BACKGROUND}. gpt-image-2 does not support transparent.",
    )
    parser.add_argument(
        "--input-fidelity",
        choices=("low", "high"),
        default=None,
        help="Optional input_fidelity value. Not sent by default.",
    )
    parser.add_argument("--user", default="", help="Optional end-user identifier for abuse monitoring.")
    parser.add_argument(
        "--rerun-existing-prompts",
        action="store_true",
        help="Generate prompts even when this output folder already contains images for that source/prompt.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without calling OpenAI.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between API calls.")
    parser.add_argument("--timeout-seconds", type=float, default=300.0, help="HTTP request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per prompt after transient failures.")

    args = parser.parse_args()
    if not args.brief_dir.exists() or not args.brief_dir.is_dir():
        raise SystemExit(f"Brief directory does not exist: {args.brief_dir}")
    if not args.api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    if not 1 <= args.images_per_prompt <= 10:
        raise SystemExit("--images-per-prompt must be between 1 and 10.")
    if args.max_images is not None and args.max_images < 1:
        raise SystemExit("--max-images must be positive when provided.")
    if args.output_compression is not None and not 0 <= args.output_compression <= 100:
        raise SystemExit("--output-compression must be between 0 and 100.")
    if args.output_compression is not None and args.output_format == "png":
        raise SystemExit("--output-compression is only supported for jpeg/webp outputs.")
    if args.model.startswith("gpt-image-2") and args.background == "transparent":
        raise SystemExit("gpt-image-2 does not support transparent backgrounds; use opaque or auto.")

    return Settings(
        brief_dir=args.brief_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        api_key=args.api_key or "",
        model=args.model,
        api_url=args.api_url.rstrip("/"),
        organization=args.organization,
        project=args.project,
        manifest_name=args.manifest_name,
        run_id=safe_name(args.run_id, "run") if args.run_id else short_id(),
        images_per_prompt=args.images_per_prompt,
        max_images=args.max_images,
        size=args.size,
        quality=args.quality,
        output_format=args.output_format,
        output_compression=args.output_compression,
        background=args.background,
        input_fidelity=args.input_fidelity,
        user=args.user,
        rerun_existing_prompts=args.rerun_existing_prompts,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_manifest_rows(settings: Settings) -> list[dict[str, Any]]:
    manifest_path = settings.brief_dir / settings.manifest_name
    if manifest_path.exists():
        rows = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}") from exc
        return rows

    brief_paths = sorted((settings.brief_dir / "briefs").glob("*.environment.json"))
    rows = []
    for brief_path in brief_paths:
        package = json.loads(brief_path.read_text(encoding="utf-8"))
        source_image = package.get("source_image")
        if not source_image:
            continue
        source_stem = Path(source_image).stem
        variant_prompts = {
            variant_id: str(settings.brief_dir / "prompts" / f"{source_stem}.{variant_id}.base_prompt.txt")
            for variant_id in package.get("variant_prompts", {})
        }
        rows.append(
            {
                "source_image": source_image,
                "brief_json": str(brief_path),
                "base_prompt": str(settings.brief_dir / "prompts" / f"{source_stem}.base_prompt.txt"),
                "variant_prompts": variant_prompts,
                "environment_type": package.get("brief", {}).get("environment_type", ""),
            }
        )
    return rows


def read_package(brief_json: Path | None) -> dict[str, Any]:
    if not brief_json or not brief_json.exists():
        return {}
    return json.loads(brief_json.read_text(encoding="utf-8"))


def read_prompt(
    prompt_path: Path | None,
    package: dict[str, Any],
    prompt_id: str,
) -> str:
    if prompt_path and prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    if prompt_id == "neutral" and isinstance(package.get("base_prompt"), str):
        return package["base_prompt"].strip()
    variant = package.get("variant_prompts", {}).get(prompt_id)
    if isinstance(variant, dict) and isinstance(variant.get("prompt"), str):
        return variant["prompt"].strip()
    raise FileNotFoundError(f"Could not find prompt text for {prompt_id}: {prompt_path}")


def ordered_variant_ids(
    variant_paths: dict[str, Any],
    package_variants: dict[str, Any],
) -> list[str]:
    ids: list[str] = []
    for variant_id in variant_paths:
        if variant_id not in ids:
            ids.append(variant_id)
    for variant_id in package_variants:
        if variant_id not in ids:
            ids.append(variant_id)
    return ids


def collect_jobs(settings: Settings) -> list[PromptJob]:
    rows = load_manifest_rows(settings)
    jobs: list[PromptJob] = []

    for row in rows:
        source_image = resolve_path(row.get("source_image"), settings.brief_dir)
        if not source_image:
            continue
        brief_json = resolve_path(row.get("brief_json"), settings.brief_dir)
        package = read_package(brief_json)
        environment_type = str(row.get("environment_type") or package.get("brief", {}).get("environment_type", ""))

        prompt_path = resolve_path(row.get("base_prompt"), settings.brief_dir)
        jobs.append(
            PromptJob(
                source_image=source_image,
                prompt_id="neutral",
                prompt_label="Neutral",
                prompt_text=read_prompt(prompt_path, package, "neutral"),
                prompt_path=prompt_path,
                brief_json=brief_json,
                environment_type=environment_type,
            )
        )

        variant_paths = row.get("variant_prompts", {})
        if not isinstance(variant_paths, dict):
            variant_paths = {}
        package_variants = package.get("variant_prompts", {})
        if not isinstance(package_variants, dict):
            package_variants = {}
        variant_ids = ordered_variant_ids(variant_paths, package_variants)
        for variant_id in variant_ids:
            prompt_path = resolve_path(variant_paths.get(variant_id), settings.brief_dir)
            label = variant_id.replace("_", " ").title()
            variant_package = package_variants.get(variant_id)
            if isinstance(variant_package, dict) and variant_package.get("label"):
                label = str(variant_package["label"])
            jobs.append(
                PromptJob(
                    source_image=source_image,
                    prompt_id=variant_id,
                    prompt_label=label,
                    prompt_text=read_prompt(prompt_path, package, variant_id),
                    prompt_path=prompt_path,
                    brief_json=brief_json,
                    environment_type=environment_type,
                )
            )

    if not jobs:
        raise SystemExit("No prompt jobs found. Check manifest.jsonl and prompt paths.")
    return jobs


def image_mime_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(image_path.name)
    if guessed in {"image/png", "image/jpeg", "image/webp"}:
        return guessed
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if image_path.suffix.lower() == ".png":
        return "image/png"
    if image_path.suffix.lower() == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported reference image type: {image_path}")


def safe_name(value: str, fallback: str = "item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:120] if cleaned else fallback


def make_output_ids(settings: Settings, count: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < count:
        output_id = f"{settings.run_id}-{short_id(6)}"
        if output_id in seen:
            continue
        seen.add(output_id)
        ids.append(output_id)
    return ids


def output_image_path(settings: Settings, job: PromptJob, output_id: str) -> Path:
    source_name = safe_name(job.source_image.stem, "source")
    prompt_name = safe_name(job.prompt_id, "prompt")
    filename = f"{source_name}.{prompt_name}.{output_id}.{settings.output_format}"
    return settings.output_dir / source_name / prompt_name / filename


def prompt_output_dirs(settings: Settings, job: PromptJob) -> list[Path]:
    source_name = safe_name(job.source_image.stem, "source")
    prompt_name = safe_name(job.prompt_id, "prompt")
    dirs = [settings.output_dir / source_name / prompt_name]
    if job.prompt_id == "neutral":
        # Backward compatibility for earlier script runs that wrote neutral prompts under "base".
        dirs.append(settings.output_dir / source_name / "base")
    return dirs


def existing_generated_images(settings: Settings, job: PromptJob) -> list[Path]:
    existing: list[Path] = []
    for directory in prompt_output_dirs(settings, job):
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_OUTPUT_SUFFIXES:
                existing.append(path)
    return existing


def output_metadata_path(image_path: Path) -> Path:
    return image_path.with_suffix(image_path.suffix + ".json")


def output_targets(settings: Settings, job: PromptJob, count: int) -> list[tuple[int, str, Path]]:
    targets = []
    for batch_index, output_id in enumerate(make_output_ids(settings, count), start=1):
        image_path = output_image_path(settings, job, output_id)
        while image_path.exists():
            output_id = f"{settings.run_id}-{short_id(6)}"
            image_path = output_image_path(settings, job, output_id)
        targets.append((batch_index, output_id, image_path))
    return targets


def multipart_body(fields: dict[str, str], image_path: Path) -> tuple[bytes, str]:
    boundary = f"----codex-openai-image-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="image[]"; '
            f'filename="{image_path.name}"\r\n'
            f"Content-Type: {image_mime_type(image_path)}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(image_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_fields(settings: Settings, job: PromptJob, n: int) -> dict[str, str]:
    fields = {
        "model": settings.model,
        "prompt": job.prompt_text,
        "n": str(n),
        "size": settings.size,
        "quality": settings.quality,
        "output_format": settings.output_format,
        "background": settings.background,
    }
    if settings.output_compression is not None:
        fields["output_compression"] = str(settings.output_compression)
    if settings.input_fidelity:
        fields["input_fidelity"] = settings.input_fidelity
    if settings.user:
        fields["user"] = settings.user
    return fields


def call_openai(settings: Settings, job: PromptJob, n: int) -> dict[str, Any]:
    url = f"{settings.api_url}/images/edits"
    fields = request_fields(settings, job, n)
    body, content_type = multipart_body(fields, job.source_image)
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": content_type,
    }
    if settings.organization:
        headers["OpenAI-Organization"] = settings.organization
    if settings.project:
        headers["OpenAI-Project"] = settings.project

    last_error: Exception | None = None
    for attempt in range(1, settings.retries + 1):
        started = time.monotonic()
        try:
            log(
                "openai request "
                f"attempt={attempt}/{settings.retries} source={job.source_image.name} "
                f"prompt={job.prompt_id} n={n} size={settings.size} "
                f"quality={settings.quality} timeout={settings.timeout_seconds:g}s"
            )
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
            elapsed = time.monotonic() - started
            log(f"openai response source={job.source_image.name} prompt={job.prompt_id} elapsed={elapsed:.1f}s")
            return json.loads(response_text)
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - started
            error_text = exc.read().decode("utf-8", errors="replace")
            if exc.code not in TRANSIENT_HTTP_CODES:
                raise RuntimeError(f"HTTP {exc.code}: {error_text[:2000]}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {error_text[:2000]}")
        except Exception as exc:  # noqa: BLE001 - CLI should report the final error.
            elapsed = time.monotonic() - started
            last_error = exc
        if attempt >= settings.retries:
            break
        retry_delay = min(2 ** (attempt - 1), 8)
        log(
            f"openai retry source={job.source_image.name} prompt={job.prompt_id} "
            f"attempt={attempt} elapsed={elapsed:.1f}s error={last_error} "
            f"sleep={retry_delay}s"
        )
        time.sleep(retry_delay)

    raise RuntimeError(f"OpenAI image request failed for {job.source_image}: {last_error}") from last_error


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_manifest(settings: Settings, row: dict[str, Any]) -> None:
    manifest_path = settings.output_dir / "generation_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_generated_images(
    settings: Settings,
    job: PromptJob,
    targets: list[tuple[int, str, Path]],
    response_json: dict[str, Any],
) -> int:
    images = response_json.get("data")
    if not isinstance(images, list):
        raise ValueError(f"OpenAI response did not contain data array: {json.dumps(response_json)[:1000]}")
    if len(images) < len(targets):
        raise ValueError(f"OpenAI returned {len(images)} images, expected {len(targets)}")

    written = 0
    created_at = datetime.now(timezone.utc).isoformat()
    for (batch_index, output_id, image_path), image_item in zip(targets, images, strict=False):
        if not isinstance(image_item, dict) or not image_item.get("b64_json"):
            raise ValueError(f"OpenAI response image is missing b64_json: {json.dumps(image_item)[:1000]}")
        image_bytes = base64.b64decode(image_item["b64_json"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        metadata_path = output_metadata_path(image_path)
        log(f"saved image={image_path}")

        metadata = {
            "source_image": str(job.source_image),
            "prompt_id": job.prompt_id,
            "prompt_label": job.prompt_label,
            "prompt_path": str(job.prompt_path) if job.prompt_path else None,
            "brief_json": str(job.brief_json) if job.brief_json else None,
            "environment_type": job.environment_type,
            "output_image": str(image_path),
            "output_id": output_id,
            "run_id": settings.run_id,
            "batch_index": batch_index,
            "model": settings.model,
            "size": settings.size,
            "quality": settings.quality,
            "output_format": settings.output_format,
            "background": settings.background,
            "api_created": response_json.get("created"),
            "revised_prompt": image_item.get("revised_prompt"),
            "created_at": created_at,
        }
        write_json(metadata_path, metadata)
        log(f"saved metadata={metadata_path}")
        append_manifest(settings, metadata)
        written += 1
    return written


def run_job(settings: Settings, job: PromptJob, targets: list[tuple[int, str, Path]]) -> int:
    if not job.source_image.exists():
        raise FileNotFoundError(f"Reference image does not exist: {job.source_image}")
    if not job.prompt_text:
        raise ValueError(f"Prompt is empty for {job.source_image} / {job.prompt_id}")

    if settings.dry_run:
        outputs = [str(image_path) for _, _, image_path in targets]
        print(
            f"would generate n={len(targets)} source={job.source_image} "
            f"prompt={job.prompt_id} outputs={outputs}",
            flush=True,
        )
        return 0

    log(f"generate n={len(targets)} source={job.source_image.name} prompt={job.prompt_id}")
    response_json = call_openai(settings, job, len(targets))
    return save_generated_images(settings, job, targets, response_json)


def main() -> int:
    settings = parse_args()
    jobs = collect_jobs(settings)

    log(f"brief_dir: {settings.brief_dir}")
    log(f"output_dir: {settings.output_dir}")
    log(f"model: {settings.model}")
    log("prompt_mode: every prompt from every image")
    log(f"prompt_jobs: {len(jobs)}")
    log(f"run_id: {settings.run_id}")
    log(f"images_per_prompt: {settings.images_per_prompt}")
    log(f"max_images: {settings.max_images if settings.max_images is not None else 'all'}")
    log(f"skip_existing_prompts: {not settings.rerun_existing_prompts}")

    remaining = settings.max_images
    generated = 0
    skipped_prompts = 0
    failed = 0

    for job in jobs:
        existing = existing_generated_images(settings, job)
        if existing and not settings.rerun_existing_prompts:
            skipped_prompts += 1
            log(
                f"skip existing source={job.source_image.name} prompt={job.prompt_id} "
                f"existing_count={len(existing)} first={existing[0]}"
            )
            continue

        count = settings.images_per_prompt
        if remaining is not None:
            if remaining <= 0:
                break
            count = min(count, remaining)
        targets = output_targets(settings, job, count)

        try:
            written = run_job(settings, job, targets)
            generated += len(targets) if settings.dry_run else written
            if remaining is not None:
                remaining -= len(targets)
            if settings.sleep_seconds > 0 and not settings.dry_run:
                time.sleep(settings.sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - continue batch after failures.
            failed += len(targets)
            print(f"ERROR: {job.source_image} / {job.prompt_id}: {exc}", file=sys.stderr, flush=True)

    log(f"done: generated_or_planned={generated}, skipped_prompts={skipped_prompts}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
