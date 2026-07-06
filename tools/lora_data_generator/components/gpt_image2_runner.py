"""GPT Image 2 pipeline: generate LoRA reference images via the OpenAI Images API.

Uses POST /v1/images/edits so the exact shared prompt from lib.prompting is sent
verbatim (the Responses API image tool would let the mainline model rewrite it,
breaking apples-to-apples comparison with Flux.2 Max). References are attached
in the same order as the Flux pipeline, so the "Reference image N" indices in
the prompt address the same files. Jobs may carry anywhere from zero up to the
maximum reference count; a job with no references falls back to
POST /v1/images/generations because the edits endpoint requires at least one
input image.

Transports:
- sync (default): one request per run; images are saved immediately.
- batch: request bodies are written to JSONL (references embedded as base64
  data URLs), uploaded, and executed by the OpenAI Batch API at 50%% token
  rates within 24h. A batch is bound to a single endpoint, so runs are grouped
  by endpoint and one batch is submitted per group (normally just one). Use
  --no-wait to submit and exit, then --fetch-batch <batch_id> later.

Output size is resolved per run: job "size", else job "width"/"height", else
--size, else --width/--height from the CLI. gpt-image-2 accepts only the sizes
in SUPPORTED_SIZES (or "auto"); anything else fails fast before any API call.

Caching: OpenAI prices a cached-input rate for gpt-image-2. To give the server
every chance to cache the repeated reference prefix, this pipeline always sends
byte-identical reference content in a stable order. The usage block returned
for every request is written to the debug log so cache hits (cached tokens) and
per-image cost can be verified from real responses rather than assumed.

Credentials: OPENAI_API_KEY, loaded from .env and resolved by main.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any

from lib.llm_client import OpenAIClient
from tool_lib.jobs import SUFFIX_BY_FORMAT, get_field, image_output_name, item_stem, normalize_items
from tool_lib.paths import load_json, resolve_repo_path, unique_path
from tool_lib.prompting import build_prompt
from tool_lib.references import ensure_extensions, ref_summary, reference_log_entries, reference_paths

MODEL = "gpt-image-2"
IMAGES_EDIT_ENDPOINT = "/v1/images/edits"
IMAGES_GENERATE_ENDPOINT = "/v1/images/generations"
DEFAULT_LOG_DIR = Path("logs/gpt_image2_lora_references")
SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}
GPT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

# USD per 1M tokens (standard tier); Batch API runs at 50% of these.
RATES_PER_1M = {"text_input": 5.0, "image_input": 8.0, "cached_input": 2.0, "image_output": 30.0}
BATCH_DISCOUNT = 0.5


def validate_size(size: str) -> str:
    if size not in SUPPORTED_SIZES and size != "auto":
        supported = ", ".join(sorted(SUPPORTED_SIZES) + ["auto"])
        raise SystemExit(
            f"gpt-image-2 only supports output sizes {supported}; got {size}. "
            "Pick a supported size (Flux.2 Max accepts arbitrary /32 dimensions, gpt-image-2 does not)."
        )
    return size


def resolve_size(item: dict[str, Any], args: argparse.Namespace) -> str:
    """Per-job size/width/height override CLI --size, which overrides --width/--height."""
    job_size = get_field(item, "size", default=None)
    if job_size:
        return validate_size(str(job_size))
    job_width = get_field(item, "width", default=None)
    job_height = get_field(item, "height", default=None)
    if job_width is not None or job_height is not None:
        return validate_size(f"{int(job_width or args.width)}x{int(job_height or args.height)}")
    if args.size:
        return validate_size(args.size)
    return validate_size(f"{args.width}x{args.height}")


def estimate_cost_usd(usage: Any, *, batch: bool) -> float | None:
    """Best-effort cost estimate from a returned usage block; logs keep the raw block."""
    if not isinstance(usage, dict):
        return None
    output_tokens = usage.get("output_tokens")
    details = usage.get("input_tokens_details") or {}
    text_in = details.get("text_tokens")
    image_in = details.get("image_tokens")
    cached = details.get("cached_tokens") or 0
    if output_tokens is None:
        return None
    if text_in is None and image_in is None:
        total_in = usage.get("input_tokens")
        if total_in is None:
            return None
        cost = total_in * RATES_PER_1M["image_input"] + output_tokens * RATES_PER_1M["image_output"]
    else:
        text_in = text_in or 0
        image_in = image_in or 0
        cached_image = min(cached, image_in)
        cost = (
            text_in * RATES_PER_1M["text_input"]
            + (image_in - cached_image) * RATES_PER_1M["image_input"]
            + cached_image * RATES_PER_1M["cached_input"]
            + output_tokens * RATES_PER_1M["image_output"]
        )
    cost /= 1_000_000
    if batch:
        cost *= BATCH_DISCOUNT
    return round(cost, 6)


def cached_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details") or {}
        value = details.get("cached_tokens")
        if isinstance(value, int):
            return value
    return 0


class RefEncoder:
    """Read and base64-encode each reference file once per process.

    Byte-identical reuse matters: the same bytes in the same order every request
    is what makes the repeated reference prefix cacheable server-side.
    """

    def __init__(self) -> None:
        self._bytes: dict[Path, bytes] = {}
        self._data_urls: dict[Path, str] = {}

    def read_bytes(self, path: Path) -> bytes:
        if path not in self._bytes:
            self._bytes[path] = path.read_bytes()
        return self._bytes[path]

    def data_url(self, path: Path) -> str:
        if path not in self._data_urls:
            mime = MIME_BY_SUFFIX[path.suffix.lower()]
            encoded = base64.b64encode(self.read_bytes(path)).decode("ascii")
            self._data_urls[path] = f"data:{mime};base64,{encoded}"
        return self._data_urls[path]


def endpoint_for_refs(refs: list[tuple[str, Path]]) -> str:
    return IMAGES_EDIT_ENDPOINT if refs else IMAGES_GENERATE_ENDPOINT


def build_json_body(run: dict[str, Any], encoder: RefEncoder, args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "prompt": run["prompt_text"],
        "size": run["size"],
        "quality": args.quality,
        "output_format": args.output_format,
        "n": 1,
    }
    if run["refs"]:
        body["image"] = [encoder.data_url(path) for _, path in run["refs"]]
    if args.moderation:
        body["moderation"] = args.moderation
    return body


def request_log_payload(
    *,
    run_index: int,
    total_runs: int,
    item_index: int,
    repeat_index: int,
    repeat_count: int,
    stem: str,
    input_stem: str,
    output_dir: Path,
    refs: list[tuple[str, Path]],
    prompt_text: str,
    size: str,
    args: argparse.Namespace,
    transport: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "total_runs": total_runs,
        "item_index": item_index,
        "repeat_index": repeat_index,
        "repeat_count": repeat_count,
        "dry_run": dry_run,
        "output_stem": stem,
        "input_json_stem": input_stem,
        "output_dir": str(output_dir),
        "model": MODEL,
        "transport": transport,
        "endpoint": endpoint_for_refs(refs),
        "size": size,
        "quality": args.quality,
        "output_format": args.output_format,
        "moderation": args.moderation,
        "references": reference_log_entries(refs),
        "prompt": prompt_text,
    }


def write_log(log_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / name
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log_path


def save_response_images(body: dict[str, Any], output_dir: Path, input_stem: str, output_format: str) -> list[Path]:
    entries = body.get("data") or []
    suffix = SUFFIX_BY_FORMAT.get(output_format, ".png")
    saved: list[Path] = []
    for entry in entries:
        b64 = entry.get("b64_json")
        if not b64:
            continue
        name = image_output_name(input_stem, "gpt", int(time.time()))
        target = unique_path(output_dir / f"{name}{suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(b64))
        saved.append(target)
    return saved


def build_runs(args: argparse.Namespace, transport: str) -> tuple[list[dict[str, Any]], int]:
    """Expand input items x repeats into run descriptors shared by sync and batch."""
    input_json = resolve_repo_path(args.input_json)
    items = normalize_items(load_json(input_json))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No prompt items found.")

    input_stem = input_json.stem
    output_dir = resolve_repo_path(args.output_dir)
    timestamp_s = int(time.time())
    total_runs = len(items) * args.repeat
    runs: list[dict[str, Any]] = []
    run_index = 0
    for index, item in enumerate(items, start=1):
        base_stem = item_stem(item, index)
        refs = reference_paths(item, input_json.parent)
        ensure_extensions(refs, GPT_IMAGE_EXTENSIONS, "gpt-image-2")
        prompt_text = build_prompt(item, refs)
        size = resolve_size(item, args)
        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            stem = f"{base_stem}_r{repeat_index:02d}" if args.repeat > 1 else base_stem
            runs.append(
                {
                    "run_index": run_index,
                    "total_runs": total_runs,
                    "item_index": index,
                    "repeat_index": repeat_index,
                    "base_stem": base_stem,
                    "stem": stem,
                    "input_stem": input_stem,
                    "output_dir": output_dir,
                    "refs": refs,
                    "prompt_text": prompt_text,
                    "size": size,
                    "log_payload": request_log_payload(
                        run_index=run_index,
                        total_runs=total_runs,
                        item_index=index,
                        repeat_index=repeat_index,
                        repeat_count=args.repeat,
                        stem=stem,
                        input_stem=input_stem,
                        output_dir=output_dir,
                        refs=refs,
                        prompt_text=prompt_text,
                        size=size,
                        args=args,
                        transport=transport,
                        dry_run=args.dry_run,
                    ),
                }
            )
    return runs, timestamp_s


def make_client(args: argparse.Namespace) -> OpenAIClient:
    return OpenAIClient(args.openai_api_key, base_url=args.base_url, timeout=args.timeout)


def run_sync(args: argparse.Namespace) -> int:
    runs, _ = build_runs(args, transport="sync")
    log_dir = resolve_repo_path(args.log_dir or DEFAULT_LOG_DIR)
    encoder = RefEncoder()
    client = make_client(args)
    total_cost = 0.0
    total_cached = 0

    for run in runs:
        refs = run["refs"]
        summary = ref_summary(refs)
        label = f"[{'dry-run ' if args.dry_run else ''}{run['run_index']}/{run['total_runs']}]"
        log_path = None
        if not args.no_log:
            log_path = write_log(log_dir, f"{run['run_index']:05d}_{run['stem']}.gpt_request.json", run["log_payload"])

        if args.dry_run:
            print(f"{label} {run['stem']}: built prompt with {summary} refs" + (f"; log: {log_path}" if log_path else ""))
            continue

        if refs:
            fields = [
                ("model", MODEL),
                ("prompt", run["prompt_text"]),
                ("size", run["size"]),
                ("quality", args.quality),
                ("output_format", args.output_format),
                ("n", "1"),
            ]
            if args.moderation:
                fields.append(("moderation", args.moderation))
            files = [
                ("image[]", path.name, encoder.read_bytes(path), MIME_BY_SUFFIX[path.suffix.lower()])
                for _, path in refs
            ]
            response = client.post_multipart(IMAGES_EDIT_ENDPOINT, fields, files)
        else:
            response = client.post_json(IMAGES_GENERATE_ENDPOINT, build_json_body(run, encoder, args))

        saved = save_response_images(response, run["output_dir"], run["input_stem"], args.output_format)
        usage = response.get("usage")
        cost = estimate_cost_usd(usage, batch=False)
        cached = cached_tokens(usage)
        total_cached += cached
        if cost is not None:
            total_cost += cost
        if log_path:
            run["log_payload"].update(
                {
                    "usage": usage,
                    "estimated_cost_usd": cost,
                    "saved_images": [str(path) for path in saved],
                }
            )
            write_log(log_dir, log_path.name, run["log_payload"])

        cost_text = f", est ${cost:.4f}" if cost is not None else ""
        cached_text = f", cached {cached} tok" if cached else ""
        names = ", ".join(path.name for path in saved) or "none"
        print(f"{label} done {run['stem']} ({summary} refs): saved {names}{cost_text}{cached_text}")

    if not args.dry_run and total_cost:
        print(f"total estimated cost: ${total_cost:.4f} (cached input tokens: {total_cached})")
    return 0


def batch_root(args: argparse.Namespace) -> Path:
    return resolve_repo_path(args.log_dir or DEFAULT_LOG_DIR) / "batches"


def run_batch(args: argparse.Namespace) -> int:
    runs, timestamp_s = build_runs(args, transport="batch")
    encoder = RefEncoder()

    # A batch is bound to one endpoint; zero-ref runs use /images/generations,
    # so group lines per endpoint (normally a single group).
    lines_by_endpoint: dict[str, list[str]] = {}
    manifest: dict[str, Any] = {}
    for run in runs:
        custom_id = f"r{run['run_index']:05d}"
        endpoint = endpoint_for_refs(run["refs"])
        body = build_json_body(run, encoder, args)
        lines_by_endpoint.setdefault(endpoint, []).append(
            json.dumps(
                {"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body},
                ensure_ascii=False,
            )
        )
        manifest[custom_id] = run["log_payload"]

    input_stem = resolve_repo_path(args.input_json).stem
    batch_dir = batch_root(args) / f"{timestamp_s}_{input_stem}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    write_log(batch_dir, "manifest.json", manifest)

    request_files: list[tuple[str, Path, bytes]] = []
    for endpoint, lines in lines_by_endpoint.items():
        suffix = "edits" if endpoint == IMAGES_EDIT_ENDPOINT else "generations"
        requests_path = batch_dir / f"requests_{suffix}.jsonl"
        requests_data = ("\n".join(lines) + "\n").encode("utf-8")
        requests_path.write_bytes(requests_data)
        request_files.append((endpoint, requests_path, requests_data))
        print(
            f"prepared {len(lines)} {suffix} request(s) "
            f"({len(requests_data) / 1_000_000:.1f} MB JSONL, references embedded as base64): {requests_path}"
        )

    if args.dry_run:
        print("[dry-run] not uploading; inspect the request JSONL and manifest.json above")
        return 0

    client = make_client(args)
    submitted_batches: list[dict[str, Any]] = []
    for endpoint, requests_path, requests_data in request_files:
        uploaded = client.upload_file(requests_path.name, requests_data, purpose="batch")
        batch = client.create_batch(
            uploaded["id"],
            endpoint,
            metadata={"source": "lora_data_generator", "input_json": input_stem},
        )
        submitted_batches.append(
            {
                "batch_id": batch["id"],
                "endpoint": endpoint,
                "input_file_id": uploaded["id"],
                "status": batch.get("status"),
                "created_at": batch.get("created_at"),
                "request_count": requests_data.count(b"\n"),
            }
        )
        print(f"submitted batch {batch['id']} ({endpoint})")
        print(f"resume later with: --mode gpt-image-2 --fetch-batch {batch['id']}")
    write_log(batch_dir, "submitted.json", {"batches": submitted_batches})
    print(f"batch state saved to {batch_dir / 'submitted.json'}")

    if args.no_wait:
        return 0
    result = 0
    for entry in submitted_batches:
        result = max(result, wait_and_fetch(client, entry["batch_id"], batch_dir, args))
    return result


def find_batch_dir(args: argparse.Namespace, batch_id: str) -> Path | None:
    root = batch_root(args)
    if not root.is_dir():
        return None
    for submitted in sorted(root.glob("*/submitted.json")):
        try:
            data = json.loads(submitted.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries = data.get("batches") or ([data] if data.get("batch_id") else [])
        if any(entry.get("batch_id") == batch_id for entry in entries):
            return submitted.parent
    return None


def wait_and_fetch(client: OpenAIClient, batch_id: str, batch_dir: Path | None, args: argparse.Namespace) -> int:
    deadline = time.time() + args.timeout
    while True:
        batch = client.get_batch(batch_id)
        status = batch.get("status")
        counts = batch.get("request_counts") or {}
        if status in TERMINAL_BATCH_STATUSES:
            break
        if time.time() >= deadline:
            print(
                f"batch {batch_id} still {status} "
                f"({counts.get('completed', 0)}/{counts.get('total', '?')} done) after --timeout; "
                f"resume with: --mode gpt-image-2 --fetch-batch {batch_id}"
            )
            return 0
        print(f"batch {batch_id}: {status}, {counts.get('completed', 0)}/{counts.get('total', '?')} done; polling...")
        time.sleep(args.poll_interval)

    if status != "completed":
        errors = batch.get("errors")
        print(f"batch {batch_id} ended as {status}" + (f"; errors: {json.dumps(errors)}" if errors else ""))
        return 1
    return fetch_results(client, batch, batch_dir, args)


def run_fetch_batch(args: argparse.Namespace) -> int:
    client = make_client(args)
    batch_dir = find_batch_dir(args, args.fetch_batch)
    if batch_dir is None:
        print(
            f"warning: no submitted.json found for {args.fetch_batch} under {batch_root(args)}; "
            "images will be named {custom_id}_gpt_{unix_seconds} in --output-dir"
        )
    batch = client.get_batch(args.fetch_batch)
    status = batch.get("status")
    if status not in TERMINAL_BATCH_STATUSES:
        counts = batch.get("request_counts") or {}
        print(f"batch {args.fetch_batch} is {status} ({counts.get('completed', 0)}/{counts.get('total', '?')} done); try again later")
        return 0
    if status != "completed":
        errors = batch.get("errors")
        print(f"batch {args.fetch_batch} ended as {status}" + (f"; errors: {json.dumps(errors)}" if errors else ""))
        return 1
    return fetch_results(client, batch, batch_dir, args)


def fetch_results(client: OpenAIClient, batch: dict[str, Any], batch_dir: Path | None, args: argparse.Namespace) -> int:
    manifest: dict[str, Any] = {}
    if batch_dir and (batch_dir / "manifest.json").is_file():
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))

    output_dir = resolve_repo_path(args.output_dir)
    saved_count = 0
    failures: list[str] = []
    total_cost = 0.0
    total_cached = 0
    results_log: dict[str, Any] = {}

    output_file_id = batch.get("output_file_id")
    if output_file_id:
        for line in client.file_content(output_file_id).decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id", "")
            entry = manifest.get(custom_id) or {}
            input_stem = entry.get("input_json_stem") or custom_id
            response = record.get("response") or {}
            body = response.get("body") or {}
            if record.get("error") or response.get("status_code") != 200:
                failures.append(f"{custom_id}: {json.dumps(record.get('error') or body)[:300]}")
                continue
            saved = save_response_images(body, output_dir, input_stem, args.output_format)
            saved_count += len(saved)
            usage = body.get("usage")
            cost = estimate_cost_usd(usage, batch=True)
            cached = cached_tokens(usage)
            total_cached += cached
            if cost is not None:
                total_cost += cost
            results_log[custom_id] = {
                "stem": entry.get("output_stem"),
                "saved_images": [str(path) for path in saved],
                "usage": usage,
                "estimated_cost_usd": cost,
            }
            stem_text = entry.get("output_stem") or custom_id
            print(f"saved {stem_text}: {len(saved)} image(s)" + (f", est ${cost:.4f}" if cost is not None else ""))

    error_file_id = batch.get("error_file_id")
    if error_file_id:
        for line in client.file_content(error_file_id).decode("utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                failures.append(f"{record.get('custom_id')}: {json.dumps(record.get('error') or record.get('response'))[:300]}")

    if batch_dir:
        write_log(batch_dir, "results.json", {"batch_id": batch.get("id"), "results": results_log, "failures": failures})

    print(
        f"batch {batch.get('id')}: saved {saved_count} image(s), {len(failures)} failure(s)"
        + (f", total est ${total_cost:.4f} (batch rates)" if total_cost else "")
        + (f", cached input tokens: {total_cached}" if total_cached else "")
    )
    for failure in failures:
        print(f"  failed {failure}")
    return 1 if failures and not saved_count else 0


def run(args: argparse.Namespace) -> int:
    if args.fetch_batch:
        return run_fetch_batch(args)
    if args.transport == "batch":
        return run_batch(args)
    return run_sync(args)
