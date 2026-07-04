#!/usr/bin/env python3
"""Describe a directory of images with Grok and write validated JSON outputs.

For every image in an input directory this script:

  1. Sends the image plus instruction prompts to a Grok vision model
     (xAI, OpenAI-compatible endpoint).
  2. Asks for this strict JSON shape:

       {
         "cinematography": "string",
         "scene": "string",
         "heroine": "string",
         "genre": "string or null",
         "flux_prompt": "string"
       }

  3. Validates the parsed response locally.
  4. If validation fails, sends the error reason back to Grok and retries.
  5. Writes ``<image_base>.json`` under ``<input_dir>/descriptions`` unless an
     explicit output directory is provided.

Auth: set the ``XAI_API_KEY`` environment variable (or pass ``--api-key``).

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Config / constants
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-4.3"  # override with --model; must be a vision-capable Grok model
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
# xAI documents jpg/jpeg/png as the reliably-supported input types and a 20 MiB cap.
XAI_SAFE_MIME = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

FIELD_CINEMATOGRAPHY = "cinematography"
FIELD_SCENE = "scene"
FIELD_HEROINE = "heroine"
FIELD_GENRE = "genre"
FIELD_FLUX_PROMPT = "flux_prompt"

REQUIRED_FIELDS = (
    FIELD_CINEMATOGRAPHY,
    FIELD_SCENE,
    FIELD_HEROINE,
    FIELD_GENRE,
    FIELD_FLUX_PROMPT,
)


class GrokValidationError(ValueError):
    """Raised when Grok returns JSON that does not match the v2 shape."""


class ValidationRetriesExhausted(RuntimeError):
    """Raised when Grok keeps returning invalid JSON after all retry attempts."""


def description_schema() -> dict[str, Any]:
    """The strict json_schema Grok must fill in."""
    required_string = {
        "type": "string",
        "minLength": 1,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            FIELD_CINEMATOGRAPHY: {
                **required_string,
                "description": "Compact camera, framing, lens, lighting, and composition description.",
            },
            FIELD_SCENE: {
                **required_string,
                "description": "Compact description of the visible setting, environment, and mood.",
            },
            FIELD_HEROINE: {
                **required_string,
                "description": "Compact description of the main visible woman, or faithfully say none is visible.",
            },
            FIELD_GENRE: {
                "type": ["string", "null"],
                "description": "A short genre label when visually inferable; otherwise null.",
            },
            FIELD_FLUX_PROMPT: {
                **required_string,
                "description": "A polished prompt-ready FLUX image generation prompt.",
            },
        },
        "required": list(REQUIRED_FIELDS),
    }


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

def image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def output_base(image: Path, input_dir: Path) -> str:
    """A collision-safe base name preserving subdirectory structure via '__'."""
    rel = image.relative_to(input_dir)
    parts = list(rel.with_suffix("").parts)
    return "__".join(parts)


def load_prompt(name: str, language: str) -> str:
    raw = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return raw.replace("{language}", language).strip()


# --------------------------------------------------------------------------- #
# Grok request
# --------------------------------------------------------------------------- #

def encode_image(image: Path) -> tuple[str, int]:
    """Return a data URI for the image and its byte size."""
    data = image.read_bytes()
    mime = MIME_BY_EXT.get(image.suffix.lower(), "application/octet-stream")
    if mime not in XAI_SAFE_MIME:
        print(
            f"  ! warning: {image.name} is {mime}; xAI reliably supports jpg/png only.",
            file=sys.stderr,
        )
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", len(data)


def initial_messages(system_prompt: str, user_prompt: str, image_data_uri: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri, "detail": "high"},
                },
            ],
        },
    ]


def retry_prompt(error_reason: str) -> str:
    return (
        "Your previous response was invalid.\n"
        f"Validation error: {error_reason}\n\n"
        "Return corrected JSON only. The response must be one JSON object with exactly these fields:\n"
        "- cinematography: non-empty string, required\n"
        "- scene: non-empty string, required\n"
        "- heroine: non-empty string, required\n"
        "- genre: string or null, required\n"
        "- flux_prompt: non-empty string, required\n"
        "Do not add commentary, markdown, or any extra fields."
    )


def build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_description_v2",
                "strict": True,
                "schema": description_schema(),
            },
        },
    }


def post_chat_completion(
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    request_retries: int,
    timeout: int,
) -> str:
    """POST to /chat/completions and return the raw assistant content string."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    attempts = max(1, request_retries)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise RuntimeError("unexpected response shape: message content is not a string")
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_err = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            last_err = RuntimeError(f"unexpected response shape: {exc}")
            break

        if attempt < attempts:
            backoff = min(2 ** attempt, 30)
            print(f"  ... request retry {attempt}/{attempts - 1} after {backoff}s ({last_err})", file=sys.stderr)
            time.sleep(backoff)

    raise RuntimeError(f"Grok request failed: {last_err}")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def parse_and_validate_response(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GrokValidationError(f"response is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise GrokValidationError("response must be a JSON object")

    expected = set(REQUIRED_FIELDS)
    unexpected = sorted(k for k in data if k not in expected)
    if unexpected:
        names = ", ".join(unexpected)
        raise GrokValidationError(f"unexpected field present: {names}")

    for field in REQUIRED_FIELDS:
        if field not in data:
            raise GrokValidationError(f"{field} field is missing, it must present")

    for field in (FIELD_CINEMATOGRAPHY, FIELD_SCENE, FIELD_HEROINE, FIELD_FLUX_PROMPT):
        value = data[field]
        if not isinstance(value, str):
            raise GrokValidationError(f"{field} field must be a string")
        if not value.strip():
            raise GrokValidationError(f"{field} field must be a non-empty string")

    genre = data[FIELD_GENRE]
    if genre is not None and not isinstance(genre, str):
        raise GrokValidationError("genre field must be a string or null")

    return {field: data[field] for field in REQUIRED_FIELDS}


def call_grok_with_validation(
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    image_data_uri: str,
    validation_retries: int,
    request_retries: int,
    timeout: int,
) -> dict[str, Any]:
    messages = initial_messages(system_prompt, user_prompt, image_data_uri)
    last_error = "unknown validation error"
    last_content = ""

    for attempt in range(0, validation_retries + 1):
        body = build_request_body(model=model, messages=messages, temperature=temperature)
        content = post_chat_completion(
            base_url=base_url,
            api_key=api_key,
            body=body,
            request_retries=request_retries,
            timeout=timeout,
        )
        last_content = content

        try:
            return parse_and_validate_response(content)
        except GrokValidationError as exc:
            last_error = str(exc)
            if attempt >= validation_retries:
                break

            retry_no = attempt + 1
            print(f"  ! invalid Grok JSON: {last_error}", file=sys.stderr)
            print(
                f"  ... validation retry {retry_no}/{validation_retries}: sending error reason back to Grok",
                file=sys.stderr,
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": retry_prompt(last_error)})

    raise ValidationRetriesExhausted(
        f"Grok returned invalid JSON after {validation_retries} validation retries: {last_error}\n"
        f"Last response:\n{last_content}"
    )


# --------------------------------------------------------------------------- #
# Per-image processing
# --------------------------------------------------------------------------- #

def process_image(
    image: Path,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Return one of: 'done', 'skipped', 'error', 'fatal'."""
    base = output_base(image, input_dir)
    json_path = output_dir / f"{base}.json"
    error_path = output_dir / f"{base}.error.txt"

    if json_path.exists() and not args.overwrite:
        print(f"= skip (exists): {image.name}")
        return "skipped"

    if error_path.exists():
        error_path.unlink()

    data_uri, size = encode_image(image)
    if size > MAX_IMAGE_BYTES:
        msg = f"{image.name} is {size / 1_048_576:.1f} MiB, over the 20 MiB xAI limit; skipping."
        print(f"  ! {msg}", file=sys.stderr)
        error_path.write_text(msg + "\n", encoding="utf-8")
        return "error"

    try:
        data = call_grok_with_validation(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_uri=data_uri,
            validation_retries=args.retries,
            request_retries=args.request_retries,
            timeout=args.timeout,
        )
    except ValidationRetriesExhausted as exc:
        print(f"  x fatal: {image.name}: {exc}", file=sys.stderr)
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        return "fatal"
    except Exception as exc:  # noqa: BLE001 - want to log and continue the batch
        print(f"  x error: {image.name}: {exc}", file=sys.stderr)
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        return "error"

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"+ done: {image.name} -> {json_path.name}")
    return "done"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Describe images with Grok and write validated v2 JSON outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir", type=Path, help="Directory of images to describe.")
    p.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: <input_dir>/descriptions).",
    )
    p.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--overwrite", action="store_true", help="Re-run images that already have output.")
    p.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all).")

    p.add_argument("--model", default=DEFAULT_MODEL, help="Vision-capable Grok model id.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="xAI OpenAI-compatible base URL.")
    p.add_argument("--api-key", default=os.environ.get("XAI_API_KEY", ""), help="xAI API key (or set XAI_API_KEY).")
    p.add_argument("--language", default="English", help="Language for Grok's description values.")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for Grok.")
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        choices=range(0, 4),
        metavar="{0,1,2,3}",
        help="Validation retries after invalid Grok JSON. Max 3.",
    )
    p.add_argument("--request-retries", type=int, default=3, help="Total attempts per HTTP request on transient errors.")
    p.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between images.")

    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not call Grok; just resolve prompts/inputs and report what would run.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir or (input_dir / "descriptions")
    output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt("grok_system.txt", args.language)
    user_prompt = load_prompt("grok_user.txt", args.language)

    images = image_paths(input_dir, args.recursive)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found in {input_dir}.", file=sys.stderr)
        return 1

    print(f"Model: {args.model}   Images: {len(images)}   Output: {output_dir}")

    if args.dry_run:
        print("\n--- DRY RUN: no Grok calls will be made ---")
        print("\n[system prompt]\n" + system_prompt)
        print("\n[user prompt]\n" + user_prompt)
        print("\n[images]")
        for img in images:
            print(f"  {img}  ->  {output_dir / (output_base(img, input_dir) + '.json')}")
        return 0

    if not args.api_key:
        print("error: no API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    counts = {"done": 0, "skipped": 0, "error": 0}
    for i, image in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image.relative_to(input_dir)}")
        result = process_image(image, input_dir, output_dir, args, system_prompt, user_prompt)
        if result == "fatal":
            print("\nExiting because Grok did not return valid JSON after all validation retries.", file=sys.stderr)
            return 1
        counts[result] += 1
        if args.sleep > 0 and i < len(images) and result != "skipped":
            time.sleep(args.sleep)

    print(f"\nDone. {counts['done']} described, {counts['skipped']} skipped, {counts['error']} errored.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
