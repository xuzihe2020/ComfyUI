#!/usr/bin/env python3
"""Caption a folder of LoRA training images with Grok.

For every image in the input directory this tool:

  1. Sends the image plus the face-identity captioning prompts to a Grok
     vision model (xAI, OpenAI-compatible endpoint).
  2. Asks for this strict JSON shape:

       {
         "caption_en": "string ({TRIGGER} placeholder kept literal)",
         "caption_zh": "string, same meaning in Chinese",
         "multiple_people": false
       }

  3. Validates the parsed response locally; on failure the error reason is
     sent back to Grok and the call is retried.
  4. Writes ``<image_base>.json`` under ``<input_dir>/captures`` (or an
     explicit ``--output-dir``). The JSON keeps the raw ``{TRIGGER}``
     placeholder and is the source of truth for captions.
  5. Writes the training caption as a sibling ``.txt`` next to the image
     (``my_image.jpeg`` -> ``my_image.txt``), with ``--trigger`` substituted.

Sibling ``.txt`` files are regenerated from existing JSON on every run, so
re-running with a different ``--trigger`` updates all captions with zero API
calls.

Auth: set the ``XAI_API_KEY`` environment variable (or pass ``--api-key``).

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lib.envfile import env_value  # noqa: E402
from lib.llm_client import GrokClient  # noqa: E402

import captioner  # noqa: E402
import dataset_io  # noqa: E402
import pricing  # noqa: E402
import run_logging  # noqa: E402

SYSTEM_PROMPT_NAME = "grok_system_caption.txt"
USER_PROMPT_NAME = "grok_user_caption.txt"


def configure_stdio() -> None:
    """Avoid Windows console crashes on non-ASCII image names/captions."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def log_request_retry(attempt: int, delay: float, err: Exception) -> None:
    print(f"  ... request retry {attempt} after {delay:.0f}s ({err})", file=sys.stderr)


def warn_txt_collisions(images: list[Path]) -> None:
    """my_face.jpg and my_face.png would fight over my_face.txt."""
    seen: dict[Path, Path] = {}
    for image in images:
        txt = dataset_io.caption_txt_path(image)
        if txt in seen:
            print(
                f"! warning: {seen[txt].name} and {image.name} share the caption file "
                f"{txt.name}; rename one of them.",
                file=sys.stderr,
            )
        seen[txt] = image


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #

def ensure_caption_json(
    image: Path,
    json_path: Path,
    error_path: Path,
    args: argparse.Namespace,
    client: GrokClient | None,
    system_prompt: str,
    user_prompt: str,
    logger: run_logging.RunLogger | None,
) -> tuple[str, dict[str, Any] | None]:
    """Return (status, data); status is 'done', 'skipped', 'error', or 'fatal'."""
    if json_path.exists() and not args.overwrite:
        try:
            data = dataset_io.read_json(json_path)
            if not isinstance(data.get(captioner.FIELD_EN), str):
                raise ValueError(f"missing {captioner.FIELD_EN} field")
        except (ValueError, json.JSONDecodeError) as exc:
            msg = f"existing {json_path.name} is unusable ({exc}); re-run with --overwrite"
            print(f"  x error: {msg}", file=sys.stderr)
            return "error", None
        print(f"= skip (exists): {json_path.name}")
        return "skipped", data

    if error_path.exists():
        error_path.unlink()

    data_uri, size = dataset_io.encode_image(image)
    if size > dataset_io.MAX_IMAGE_BYTES:
        msg = f"{image.name} is {size / 1_048_576:.1f} MiB, over the 20 MiB xAI limit; skipping."
        print(f"  ! {msg}", file=sys.stderr)
        error_path.write_text(msg + "\n", encoding="utf-8")
        return "error", None

    log_attempt = None
    if logger is not None:
        def log_attempt(**kwargs: Any) -> None:
            usage = logger.log_attempt(image=image.name, **kwargs)
            if kwargs.get("response") is None:
                return  # transport failure: no usage to report, error is printed elsewhere
            cost = usage["est_cost_usd"]
            cost_str = f", ~${cost:.4f}" if cost is not None else ""
            print(
                f"  request {kwargs['attempt']}: {usage['input_tokens']} in / "
                f"{usage['output_tokens']} out{cost_str}"
            )

    try:
        data = captioner.call_grok_with_validation(
            client,
            model=args.model,
            temperature=args.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_uri=data_uri,
            validation_retries=args.retries,
            request_retries=args.request_retries,
            log_attempt=log_attempt,
            on_request_retry=log_request_retry,
        )
    except captioner.ValidationRetriesExhausted as exc:
        print(f"  x fatal: {image.name}: {exc}", file=sys.stderr)
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        return "fatal", None
    except Exception as exc:  # noqa: BLE001 - log and continue the batch
        print(f"  x error: {image.name}: {exc}", file=sys.stderr)
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        return "error", None

    data = {"image": image.name, **data, "model": args.model}
    dataset_io.write_json(json_path, data)
    print(f"+ done: {image.name} -> {json_path.name}")
    return "done", data


def print_usage_summary(logger: run_logging.RunLogger | None, prices: tuple[float, float] | None) -> None:
    if logger is None:
        return
    totals = logger.close()
    cost_str = f"~${totals['cost_usd']:.4f}" if prices is not None else "n/a (unknown model pricing)"
    print(
        f"Grok usage: {totals['requests']} request(s), "
        f"{totals['input_tokens']} input tokens, {totals['output_tokens']} output tokens, "
        f"est cost {cost_str}"
    )
    if logger.path is not None:
        print(f"Request log: {logger.path}")


def write_caption_txt(image: Path, data: dict[str, Any], trigger: str) -> None:
    txt_path = dataset_io.caption_txt_path(image)
    text = captioner.render_caption_txt(data[captioner.FIELD_EN], trigger)
    changed = dataset_io.write_text_if_changed(txt_path, text)
    print(f"  txt: {txt_path.name} ({'updated' if changed else 'unchanged'})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caption LoRA training images with Grok: JSON per image + sibling .txt captions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", type=Path, help="Directory of training images to caption.")
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Caption JSON directory (default: <input_dir>/captures).",
    )
    p.add_argument(
        "--trigger", default="",
        help="Trigger token substituted for {TRIGGER} in the sibling .txt captions. "
             "When omitted, .txt files keep the literal {TRIGGER} placeholder.",
    )
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--overwrite", action="store_true", help="Re-caption images that already have JSON.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all).")

    p.add_argument("--model", default=GrokClient.DEFAULT_MODEL, help="Vision-capable Grok model id.")
    p.add_argument("--base-url", default=GrokClient.DEFAULT_BASE_URL, help="xAI OpenAI-compatible base URL.")
    p.add_argument(
        "--api-key",
        default="",
        help="xAI API key, or set XAI_API_KEY in the environment or repo .env file.",
    )
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for Grok.")
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        choices=range(0, 4),
        metavar="{0,1,2,3}",
        help="Validation retries after invalid Grok captions. Max 3.",
    )
    p.add_argument("--request-retries", type=int, default=3, help="Total attempts per HTTP request on transient errors.")
    p.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between captioned images.")
    p.add_argument(
        "--logs-dir", type=Path, default=None,
        help="Directory for per-run request logs (default: <input_dir>/logs).",
    )
    p.add_argument(
        "--price-input", type=float, default=None,
        help="Override input price in USD per million tokens for cost estimates.",
    )
    p.add_argument(
        "--price-output", type=float, default=None,
        help="Override output price in USD per million tokens for cost estimates.",
    )
    p.add_argument(
        "--debug",
        type=int,
        choices=(0, 1),
        default=0,
        help="When 1, write raw request/response JSONL logs to the logs dir.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not call Grok; just resolve prompts/inputs and report what would run.",
    )
    args = p.parse_args(argv)
    if not args.api_key:
        args.api_key = env_value("XAI_API_KEY")
    return args


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir or (input_dir / "captures")
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = dataset_io.load_prompt(SYSTEM_PROMPT_NAME)
    user_prompt = dataset_io.load_prompt(USER_PROMPT_NAME)
    logs_dir: Path = args.logs_dir or (input_dir / "logs")
    prices = pricing.resolve_prices(args.model, args.price_input, args.price_output)

    images = dataset_io.image_paths(input_dir, args.recursive)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"No images ({', '.join(sorted(dataset_io.IMAGE_EXTS))}) found in {input_dir}.", file=sys.stderr)
        return 1

    warn_txt_collisions(images)

    json_path_for = {
        image: output_dir / f"{dataset_io.output_base(image, input_dir)}.json" for image in images
    }
    pending = [image for image in images if args.overwrite or not json_path_for[image].exists()]

    print(
        f"Model: {args.model}   Images: {len(images)} ({len(pending)} need Grok)   "
        f"Output: {output_dir}   Trigger: {args.trigger or '<placeholder kept>'}"
    )
    logs_note = f"Debug logs: {logs_dir}" if args.debug else "Debug logs: off (--debug 1 to enable)"
    if prices is not None:
        print(f"Pricing: ${prices[0]:.2f}/M input, ${prices[1]:.2f}/M output   {logs_note}")
    else:
        print(
            f"Pricing: unknown for model {args.model} — costs shown as n/a "
            f"(set --price-input/--price-output)   {logs_note}"
        )

    if args.dry_run:
        print("\n--- DRY RUN: no Grok calls will be made ---")
        print("\n[system prompt]\n" + system_prompt)
        print("\n[user prompt]\n" + user_prompt)
        print("\n[images]")
        for image in images:
            marker = "grok" if image in pending else "cached"
            print(f"  {image}  ->  {json_path_for[image]}  +  {dataset_io.caption_txt_path(image)}  [{marker}]")
        return 0

    client: GrokClient | None = None
    logger: run_logging.RunLogger | None = None
    if pending:
        if not args.api_key:
            print("error: no API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
            return 2
        client = GrokClient(args.api_key, base_url=args.base_url, timeout=args.timeout)
        logger = run_logging.RunLogger(
            logs_dir if args.debug else None, model=args.model, prices=prices,
        )

    counts = {"done": 0, "skipped": 0, "error": 0}
    flagged: list[str] = []
    for i, image in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image.relative_to(input_dir)}")
        base = dataset_io.output_base(image, input_dir)
        status, data = ensure_caption_json(
            image=image,
            json_path=json_path_for[image],
            error_path=output_dir / f"{base}.error.txt",
            args=args,
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            logger=logger,
        )
        if status == "fatal":
            print("\nExiting because Grok did not return valid captions after all validation retries.", file=sys.stderr)
            print_usage_summary(logger, prices)
            return 1
        counts[status] += 1
        if data is None:
            continue

        write_caption_txt(image, data, args.trigger)
        if data.get(captioner.FIELD_MULTI):
            flagged.append(str(image.relative_to(input_dir)))
            print(f"  ! flagged multiple_people: {image.name}", file=sys.stderr)

        if args.sleep > 0 and i < len(images) and status == "done":
            time.sleep(args.sleep)

    print(f"\nDone. {counts['done']} captioned, {counts['skipped']} reused, {counts['error']} errored.")
    print_usage_summary(logger, prices)
    if flagged:
        print(
            f"! {len(flagged)} image(s) flagged multiple_people — cull or re-crop before training:",
            file=sys.stderr,
        )
        for name in flagged:
            print(f"    {name}", file=sys.stderr)
    if not args.trigger:
        print(
            "note: no --trigger given; .txt captions keep the literal {TRIGGER} placeholder. "
            "Re-run with --trigger <token> before training (cached JSON is reused, no API cost).",
        )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
