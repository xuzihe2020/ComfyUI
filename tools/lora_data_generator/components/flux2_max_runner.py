"""Flux.2 Max pipeline: generate LoRA reference images via the official BFL API.

Calls POST /v1/flux-2-max on api.bfl.ai directly — no ComfyUI involved — so
FLUX_API_KEY is a Black Forest Labs key from https://dashboard.bfl.ai.

Prompt text and reference ordering come from lib.prompting / lib.references,
shared with the GPT Image 2 pipeline so both models receive identical inputs:
references are attached as input_image, input_image_2, ... in exactly the order
the prompt's "Attached reference image order" block describes (BFL accepts up
to 8 input images; this pipeline's 2 dressing + 5 character maximum fits).

Requests are asynchronous: submit, poll the returned polling_url until Ready,
then download the signed result URL (valid ~10 minutes) into --output-dir.
The submit response reports the actual billed cost and input/output megapixels,
which are printed and written to the per-request debug log — no cost estimating
needed on this pipeline.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
from pathlib import Path
from typing import Any

from components.bfl_client import BFLClient
from lib.jobs import SUFFIX_BY_FORMAT, get_field, image_output_name, item_stem, normalize_items
from lib.paths import load_json, resolve_repo_path, unique_path
from lib.prompting import build_prompt
from lib.references import ensure_extensions, ref_summary, reference_log_entries, reference_paths

BFL_ENDPOINT = "/v1/flux-2-max"
DEFAULT_BASE_URL = "https://api.bfl.ai"
DEFAULT_LOG_DIR = Path("logs/flux2_max_lora_references")
FLUX_API_KEY_ENV_KEYS = ("FLUX_API_KEY", "BFL_API_KEY")
BFL_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_INPUT_IMAGES = 8
MIN_DIMENSION = 64
MAX_SEED = 2**31 - 1
POLL_INTERVAL_S = 2.0


def validate_dimension(name: str, value: int) -> int:
    if value < MIN_DIMENSION:
        raise ValueError(f"{name} must be at least {MIN_DIMENSION}; got {value}")
    return value


def encode_ref(path: Path, cache: dict[Path, str]) -> str:
    """Base64-encode each reference file once per process, byte-identical across runs."""
    if path not in cache:
        cache[path] = base64.b64encode(path.read_bytes()).decode("ascii")
    return cache[path]


def attach_references(payload: dict[str, Any], refs: list[tuple[str, Path]], cache: dict[Path, str]) -> None:
    if len(refs) > MAX_INPUT_IMAGES:
        raise ValueError(f"BFL accepts at most {MAX_INPUT_IMAGES} input images; got {len(refs)}")
    for index, (_, path) in enumerate(refs, start=1):
        field = "input_image" if index == 1 else f"input_image_{index}"
        payload[field] = encode_ref(path, cache)


def write_log(log_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / name
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log_path


def run(args: argparse.Namespace) -> int:
    # API keys are loaded from .env and resolved by main.py.
    client = BFLClient(args.flux_api_key, base_url=args.bfl_base_url)

    input_json = resolve_repo_path(args.input_json)
    items = normalize_items(load_json(input_json))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No prompt items found.")

    log_dir = resolve_repo_path(args.log_dir or DEFAULT_LOG_DIR)
    output_dir = resolve_repo_path(args.output_dir)
    suffix = SUFFIX_BY_FORMAT.get(args.output_format, ".png")
    encode_cache: dict[Path, str] = {}
    total_runs = len(items) * args.repeat
    total_cost = 0.0
    run_index = 0

    for index, item in enumerate(items, start=1):
        base_stem = item_stem(item, index)
        refs = reference_paths(item, input_json.parent)
        ensure_extensions(refs, BFL_IMAGE_EXTENSIONS, "the BFL API")
        prompt_text = build_prompt(item, refs)
        summary = ref_summary(refs)
        width = validate_dimension("width", int(get_field(item, "width", default=args.width)))
        height = validate_dimension("height", int(get_field(item, "height", default=args.height)))

        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            stem = f"{base_stem}_r{repeat_index:02d}" if args.repeat > 1 else base_stem
            seed_value = get_field(item, "seed", default=args.seed)
            seed = random.randint(0, MAX_SEED) if seed_value is None else int(seed_value)

            log_payload: dict[str, Any] = {
                "run_index": run_index,
                "total_runs": total_runs,
                "item_index": index,
                "repeat_index": repeat_index,
                "repeat_count": args.repeat,
                "dry_run": args.dry_run,
                "output_stem": stem,
                "input_json_stem": input_json.stem,
                "output_dir": str(output_dir),
                "model": "flux-2-max",
                "endpoint": BFL_ENDPOINT,
                "width": width,
                "height": height,
                "seed": seed,
                "output_format": args.output_format,
                "safety_tolerance": args.safety_tolerance,
                "references": reference_log_entries(refs),
                "prompt": prompt_text,
            }
            log_path = None
            if not args.no_log:
                log_path = write_log(log_dir, f"{run_index:05d}_{stem}.flux2_request.json", log_payload)

            repeat_text = f" repeat {repeat_index}/{args.repeat}" if args.repeat > 1 else ""
            log_text = f"; log: {log_path}" if log_path else ""
            if args.dry_run:
                print(f"[dry-run {run_index}/{total_runs}] {base_stem}{repeat_text}: built prompt with {summary} refs{log_text}")
                continue

            payload: dict[str, Any] = {
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "seed": seed,
                "output_format": args.output_format,
            }
            if args.safety_tolerance is not None:
                payload["safety_tolerance"] = args.safety_tolerance
            attach_references(payload, refs, encode_cache)

            submitted = client.post_json(BFL_ENDPOINT, payload)
            request_id = submitted.get("id")
            cost = submitted.get("cost")
            if isinstance(cost, (int, float)):
                total_cost += cost
            print(
                f"[{run_index}/{total_runs}] submitted {base_stem}{repeat_text} ({summary} refs): {request_id}"
                + (f", cost {cost}" if cost is not None else "")
                + (
                    f" (input {submitted.get('input_mp')} MP, output {submitted.get('output_mp')} MP)"
                    if submitted.get("input_mp") is not None
                    else ""
                )
            )

            result = client.poll_until_ready(submitted["polling_url"], args.timeout, POLL_INTERVAL_S)
            sample_url = (result.get("result") or {}).get("sample")
            if not sample_url:
                raise RuntimeError(f"BFL task {request_id} is Ready but has no result.sample: {json.dumps(result)[:300]}")
            data = client.download(sample_url)
            name = image_output_name(input_json.stem, "flux2", int(time.time()))
            target = unique_path(output_dir / f"{name}{suffix}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

            if log_path:
                log_payload.update(
                    {
                        "request_id": request_id,
                        "cost": cost,
                        "input_mp": submitted.get("input_mp"),
                        "output_mp": submitted.get("output_mp"),
                        "saved_images": [str(target)],
                    }
                )
                write_log(log_dir, log_path.name, log_payload)
            print(f"[{run_index}/{total_runs}] done {base_stem}{repeat_text}: saved {target.name} in {output_dir}{log_text}")

    if not args.dry_run and total_cost:
        print(f"total billed cost reported by BFL: {round(total_cost, 6)}")
    return 0
