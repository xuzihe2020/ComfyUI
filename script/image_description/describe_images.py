#!/usr/bin/env python3
"""Describe a directory of images with Grok, then build FLUX.2 repaint prompts.

For every image in an input directory this script:

  1. Sends the image plus an instruction prompt to a Grok vision model
     (xAI, OpenAI-compatible endpoint) and asks for a detailed description
     returned as strict structured JSON.
  2. Writes that JSON object to ``<base>.json``.
  3. Deterministically assembles a FLUX.2 text-to-image prompt from the JSON
     (with optional user-supplied prefix / suffix blocks) and writes it to
     ``<base>.flux2.txt``.

The prompts sent to Grok live as plain-text blobs under ``prompts/`` so they can
be reviewed and edited without touching this code. The FLUX.2 prompt is built by
plain, static string assembly from the JSON fields -- no model call is involved.

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

# JSON schema keys, in the order we want them to flow into the FLUX.2 prompt.
FIELD_CINEMATOGRAPHY = "cinematography"
FIELD_SCENE = "scene"
FIELD_HEROINE = "heroine"

CIN_PERSPECTIVE = "perspective"
CIN_SHOT_SIZE = "shot_size"
CIN_FOCUS_TYPE = "focus_type"
CIN_BODY_PART = "body_part"
CIN_ANGLE = "angle"
CIN_COMPOSITION_NOTES = "composition_notes"

SCENE_LOCATION = "location"
SCENE_TIME = "time"
SCENE_LIGHTING = "lighting"
SCENE_ENVIRONMENT = "environment"

HEROINE_NAME = "name"
HEROINE_PROPORTION = "proportion_in_frame"
HEROINE_BODY = "body"
HEROINE_FACE = "face"
HEROINE_HAIRSTYLE = "hairstyle"
HEROINE_CLOTHING = "clothing_state"
HEROINE_EXPRESSION = "expression"
HEROINE_ACTION = "body_action"
HEROINE_CAMERA = "relationship_to_camera"

PERSPECTIVE_TEXT = {
    "first_person_male_protagonist_pov": "first person POV looking down",
    "third_person_side": "third person side view, no male face visible",
    "third_person_over_shoulder": "over the shoulder view from behind the man",
    "third_person_back": "from behind the man, only his back visible",
}

SHOT_SIZE_TEXT = {
    "full_body": "full body shot",
    "medium_full": "medium full shot (knees up)",
    "medium": "medium shot (waist up)",
    "medium_close_up": "medium close-up (chest up)",
    "close_up": "close-up",
    "extreme_close_up": "extreme close-up",
}

ANGLE_TEXT = {
    "eye_level": "eye level",
    "low_angle": "low angle",
    "high_angle": "high angle",
    "dutch_angle": "dutch angle",
}

FOCUS_TYPE_TEXT = {
    "character": "",
    "body_part_closeup": "extreme close-up of body part, no face visible",
}

DEFAULT_PERSPECTIVE = "first_person_male_protagonist_pov"
DEFAULT_SHOT_SIZE = "medium"
DEFAULT_ANGLE = "eye_level"
DEFAULT_FOCUS_TYPE = "character"

QUALITY_PROMPT = (
    "photorealistic, ultra-detailed skin texture, cinematic lighting, sharp focus, "
    "8k, masterpiece, best quality --ar 16:9"
)


def description_schema() -> dict[str, Any]:
    """The strict json_schema Grok must fill in."""
    nullable_string = {"type": ["string", "null"]}
    cinematography = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            CIN_PERSPECTIVE: {
                "type": "string",
                "enum": list(PERSPECTIVE_TEXT),
                "description": "Camera viewpoint enum. Choose the closest match; enum value only.",
            },
            CIN_SHOT_SIZE: {
                "type": "string",
                "enum": list(SHOT_SIZE_TEXT),
                "description": "Shot scale enum. Choose the closest match; enum value only.",
            },
            CIN_FOCUS_TYPE: {
                "type": "string",
                "enum": list(FOCUS_TYPE_TEXT),
                "description": "Use body_part_closeup only when the image clearly focuses on one body part.",
            },
            CIN_BODY_PART: {
                **nullable_string,
                "description": "1-4 words naming the focused body part, or null when focus_type is character.",
            },
            CIN_ANGLE: {
                "type": "string",
                "enum": list(ANGLE_TEXT),
                "description": "Camera angle enum. Choose the closest match; enum value only.",
            },
            CIN_COMPOSITION_NOTES: {
                "type": "string",
                "description": "6-18 words on framing, crop, occlusion, leading lines, or important composition.",
            },
        },
        "required": [
            CIN_PERSPECTIVE,
            CIN_SHOT_SIZE,
            CIN_FOCUS_TYPE,
            CIN_BODY_PART,
            CIN_ANGLE,
            CIN_COMPOSITION_NOTES,
        ],
    }
    scene = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            SCENE_LOCATION: {
                "type": "string",
                "description": "3-10 words naming the visible setting or place.",
            },
            SCENE_TIME: {
                "type": "string",
                "description": "1-5 words for visible or implied time of day.",
            },
            SCENE_LIGHTING: {
                "type": "string",
                "description": "4-12 words on light source, direction, color, hardness, or mood.",
            },
            SCENE_ENVIRONMENT: {
                "type": "string",
                "description": "6-18 words describing background, props, surfaces, weather, or atmosphere.",
            },
        },
        "required": [SCENE_LOCATION, SCENE_TIME, SCENE_LIGHTING, SCENE_ENVIRONMENT],
    }
    heroine = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            HEROINE_NAME: {
                "type": "string",
                "description": "1-4 words identifying the main woman; use 'the woman' if unnamed.",
            },
            HEROINE_PROPORTION: {
                "type": "string",
                "description": "4-10 words for her scale and placement in frame.",
            },
            HEROINE_BODY: {
                "type": "string",
                "description": "6-16 words for age range, build, pose, skin tone, and visible body traits.",
            },
            HEROINE_FACE: {
                **nullable_string,
                "description": "6-14 words for face shape/features if visible, otherwise null.",
            },
            HEROINE_HAIRSTYLE: {
                **nullable_string,
                "description": "3-10 words for hair color, length, and style if visible, otherwise null.",
            },
            HEROINE_CLOTHING: {
                "type": "string",
                "description": "6-16 words for garments, colors, materials, fit, accessories, and footwear.",
            },
            HEROINE_EXPRESSION: {
                **nullable_string,
                "description": "3-10 words for visible emotion or facial expression, otherwise null.",
            },
            HEROINE_ACTION: {
                "type": "string",
                "description": "6-16 words for pose, gesture, body orientation, and current action.",
            },
            HEROINE_CAMERA: {
                "type": "string",
                "description": "5-14 words for facing direction, eye contact, distance, and camera relation.",
            },
        },
        "required": [
            HEROINE_NAME,
            HEROINE_PROPORTION,
            HEROINE_BODY,
            HEROINE_FACE,
            HEROINE_HAIRSTYLE,
            HEROINE_CLOTHING,
            HEROINE_EXPRESSION,
            HEROINE_ACTION,
            HEROINE_CAMERA,
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            FIELD_CINEMATOGRAPHY: cinematography,
            FIELD_SCENE: scene,
            FIELD_HEROINE: heroine,
        },
        "required": [FIELD_CINEMATOGRAPHY, FIELD_SCENE, FIELD_HEROINE],
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


def read_text_arg(value: str | None, path: str | None, label: str) -> str:
    """Resolve a text block from either an inline string or a file path."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    elif value is not None:
        text = value
    else:
        text = ""
    return text.strip()


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


def build_request_body(
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_uri: str,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
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
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_description",
                "strict": True,
                "schema": description_schema(),
            },
        },
    }


def call_grok(
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    """POST to /chat/completions with retry/backoff; return the parsed JSON content."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            last_err = RuntimeError(f"HTTP {exc.code}: {detail}")
            # Retry only on rate limit / server errors.
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            last_err = RuntimeError(f"unexpected response shape: {exc}")
            break

        if attempt < retries:
            backoff = min(2 ** attempt, 30)
            print(f"  ... retry {attempt}/{retries - 1} after {backoff}s ({last_err})", file=sys.stderr)
            time.sleep(backoff)

    raise RuntimeError(f"Grok request failed: {last_err}")


# --------------------------------------------------------------------------- #
# FLUX.2 prompt construction (static, deterministic)
# --------------------------------------------------------------------------- #

def _clean(text: Any) -> str:
    """Trim and drop trailing punctuation so prompt lines format cleanly."""
    s = str(text or "").strip()
    while s.endswith((".", ";", ",")):
        s = s[:-1].strip()
    return s


def _enum_text(mapping: dict[str, str], enum_value: object, default_enum_value: str) -> str:
    return mapping.get(enum_value, mapping[default_enum_value])


def _prompt_lines(parts: list[object]) -> list[str]:
    return [text for part in parts if (text := _clean(part))]


def _format_prompt_block(parts: list[object]) -> str:
    lines = _prompt_lines(parts)
    if not lines:
        return ""

    punctuated = [f"{line}," for line in lines[:-1]]
    punctuated.append(f"{lines[-1]}.")
    return "\n".join(punctuated)


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(block for block in blocks if block)


def build_scene_prompt_block(scene: dict[str, Any]) -> str:
    return _format_prompt_block(
        [
            scene.get(SCENE_LOCATION),
            scene.get(SCENE_TIME),
            scene.get(SCENE_LIGHTING),
            scene.get(SCENE_ENVIRONMENT),
        ]
    )


def build_cinematography_prompt_block(cinematography: dict[str, Any]) -> str:
    perspective = cinematography.get(CIN_PERSPECTIVE, DEFAULT_PERSPECTIVE)
    shot_size = cinematography.get(CIN_SHOT_SIZE, DEFAULT_SHOT_SIZE)
    focus_type = cinematography.get(CIN_FOCUS_TYPE, DEFAULT_FOCUS_TYPE)
    angle = cinematography.get(CIN_ANGLE, DEFAULT_ANGLE)

    return _format_prompt_block(
        [
            _enum_text(PERSPECTIVE_TEXT, perspective, DEFAULT_PERSPECTIVE),
            _enum_text(SHOT_SIZE_TEXT, shot_size, DEFAULT_SHOT_SIZE),
            _enum_text(FOCUS_TYPE_TEXT, focus_type, DEFAULT_FOCUS_TYPE),
            cinematography.get(CIN_BODY_PART),
            _enum_text(ANGLE_TEXT, angle, DEFAULT_ANGLE),
            cinematography.get(CIN_COMPOSITION_NOTES),
        ]
    )


def build_heroine_prompt_block(heroine: dict[str, Any], cinematography: dict[str, Any]) -> str:
    focus_type = cinematography.get(CIN_FOCUS_TYPE, DEFAULT_FOCUS_TYPE)
    if focus_type == "body_part_closeup":
        return _format_prompt_block(
            [
                heroine.get(HEROINE_CLOTHING),
                heroine.get(HEROINE_ACTION),
                heroine.get(HEROINE_CAMERA),
            ]
        )

    return _format_prompt_block(
        [
            heroine.get(HEROINE_NAME),
            heroine.get(HEROINE_PROPORTION),
            heroine.get(HEROINE_BODY),
            heroine.get(HEROINE_FACE),
            heroine.get(HEROINE_HAIRSTYLE),
            heroine.get(HEROINE_CLOTHING),
            heroine.get(HEROINE_EXPRESSION),
            heroine.get(HEROINE_ACTION),
            heroine.get(HEROINE_CAMERA),
        ]
    )


def build_flux2_prompt(
    data: dict[str, Any],
    prefix: str,
    suffix: str,
) -> str:
    """Assemble a FLUX.2 prompt from the structured description.

    FLUX.2 favors compact natural-language descriptors and explicit scene,
    camera, and character detail. We keep comma-separated lines inside each
    block, end each block with a period, and separate larger blocks by a blank
    line so the prompt remains readable without becoming wordy.
    """
    cinematography = data.get(FIELD_CINEMATOGRAPHY) or {}
    scene = data.get(FIELD_SCENE) or {}
    heroine = data.get(FIELD_HEROINE) or {}

    body = _join_blocks(
        [
            build_scene_prompt_block(scene),
            build_cinematography_prompt_block(cinematography),
            build_heroine_prompt_block(heroine, cinematography),
            QUALITY_PROMPT,
        ]
    )

    blocks = [b for b in (prefix.strip(), body, suffix.strip()) if b]
    return "\n\n".join(blocks) + "\n"


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
    prefix: str,
    suffix: str,
) -> str:
    """Return one of: 'done', 'skipped', 'error'."""
    base = output_base(image, input_dir)
    json_path = output_dir / f"{base}.json"
    flux_path = output_dir / f"{base}.flux2.txt"
    error_path = output_dir / f"{base}.error.txt"

    if json_path.exists() and flux_path.exists() and not args.overwrite:
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

    body = build_request_body(
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_data_uri=data_uri,
        temperature=args.temperature,
    )

    try:
        data = call_grok(
            base_url=args.base_url,
            api_key=args.api_key,
            body=body,
            retries=args.retries,
            timeout=args.timeout,
        )
    except Exception as exc:  # noqa: BLE001 - want to log and continue the batch
        print(f"  x error: {image.name}: {exc}", file=sys.stderr)
        error_path.write_text(f"{exc}\n", encoding="utf-8")
        return "error"

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flux_prompt = build_flux2_prompt(data, prefix, suffix)
    flux_path.write_text(flux_prompt, encoding="utf-8")
    print(f"+ done: {image.name} -> {json_path.name}, {flux_path.name}")
    return "done"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Describe images with Grok and build FLUX.2 repaint prompts.",
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
    p.add_argument("--retries", type=int, default=4, help="Total attempts per image on transient errors.")
    p.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between images.")

    p.add_argument("--prefix", default=None, help="Text prepended to every FLUX.2 prompt.")
    p.add_argument("--prefix-file", default=None, help="File whose contents are prepended (overrides --prefix).")
    p.add_argument("--suffix", default=None, help="Text appended to every FLUX.2 prompt.")
    p.add_argument("--suffix-file", default=None, help="File whose contents are appended (overrides --suffix).")

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

    prefix = read_text_arg(args.prefix, args.prefix_file, "prefix")
    suffix = read_text_arg(args.suffix, args.suffix_file, "suffix")

    system_prompt = load_prompt("grok_system.txt", args.language)
    user_prompt = load_prompt("grok_user.txt", args.language)

    images = image_paths(input_dir, args.recursive)
    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        print(f"No images ({', '.join(sorted(IMAGE_EXTS))}) found in {input_dir}.", file=sys.stderr)
        return 1

    # Record exactly what is being sent to Grok, for review/debugging.
    (output_dir / "_effective_grok_system.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (output_dir / "_effective_grok_user.txt").write_text(user_prompt + "\n", encoding="utf-8")
    (output_dir / "_run_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_url": args.base_url,
                "language": args.language,
                "temperature": args.temperature,
                "input_dir": str(input_dir),
                "recursive": args.recursive,
                "image_count": len(images),
                "prefix": prefix,
                "suffix": suffix,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Model: {args.model}   Images: {len(images)}   Output: {output_dir}")

    if args.dry_run:
        print("\n--- DRY RUN: no Grok calls will be made ---")
        print("\n[system prompt]\n" + system_prompt)
        print("\n[user prompt]\n" + user_prompt)
        print(f"\n[FLUX.2 prefix]\n{prefix or '(empty)'}")
        print(f"\n[FLUX.2 suffix]\n{suffix or '(empty)'}")
        print("\n[images]")
        for img in images:
            print(f"  {img}  ->  {output_base(img, input_dir)}.{{json,flux2.txt}}")
        return 0

    if not args.api_key:
        print("error: no API key. Set XAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    counts = {"done": 0, "skipped": 0, "error": 0}
    for i, image in enumerate(images, start=1):
        print(f"[{i}/{len(images)}] {image.relative_to(input_dir)}")
        result = process_image(
            image, input_dir, output_dir, args, system_prompt, user_prompt, prefix, suffix
        )
        counts[result] += 1
        if args.sleep > 0 and i < len(images) and result != "skipped":
            time.sleep(args.sleep)

    print(f"\nDone. {counts['done']} described, {counts['skipped']} skipped, {counts['error']} errored.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
