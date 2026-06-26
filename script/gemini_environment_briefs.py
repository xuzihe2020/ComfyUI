#!/usr/bin/env python3
"""Batch-extract VN environment briefs from image references using Gemini.

The script scans an image directory, sends each image to the Gemini API, asks
for a strict JSON environment brief, and renders a base-image prompt from that
brief. It does not call ComfyUI nodes directly; the outputs are plain JSON/TXT
files that can be consumed by ComfyUI workflows or prompt templates.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_RENDERING_STYLE = (
    "3D Unity CG rendering style, like a game environment rendered in Unity, "
    "clean realtime 3D scene, coherent geometry, polished game-environment "
    "materials, soft baked lighting, not photorealistic"
)

CORE_FIELDS = (
    "environment_type",
    "scene_location",
    "has_windows",
    "style_description",
    "required_items",
    "character_safe_zone",
    "identity_anchors",
    "avoid",
)

OPTIONAL_FIELDS = (
    "lighting",
    "color_palette",
    "materials",
    "layout_summary",
)

BRIEF_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS
LIST_FIELDS = {"color_palette", "materials", "required_items", "identity_anchors", "avoid"}
PROMPT_LIST_LIMITS = {
    "required_items": 8,
    "identity_anchors": 6,
    "avoid": 8,
}

ENVIRONMENT_BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "environment_type": {"type": "string"},
        "scene_location": {"type": "string", "enum": ["indoor", "outdoor"]},
        "has_windows": {"type": "string", "enum": ["yes", "no", "not_applicable", "unclear"]},
        "style_description": {"type": "string"},
        "lighting": {"type": "string"},
        "color_palette": {"type": "array", "items": {"type": "string"}},
        "materials": {"type": "array", "items": {"type": "string"}},
        "required_items": {"type": "array", "items": {"type": "string"}},
        "layout_summary": {"type": "string"},
        "character_safe_zone": {"type": "string"},
        "identity_anchors": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
    },
    "required": list(CORE_FIELDS),
}


EXTRACTION_PROMPT = """Analyze this image as a visual novel background reference.
Return a concise structured brief. Do not invent characters. Focus on the
environment, visual style, layout, materials, lighting, and objects that should
be preserved when generating an extended visual novel background.
Do not describe the rendering medium or art style; the target rendering style
is fixed by the prompt template. Describe only what is in the scene and its
environment design language.
Ignore any watermark, signature, stock-photo mark, UI overlay, border, caption,
or text overlay in the reference image. These are not part of the environment
and should not be preserved.

Return only JSON matching the provided schema. Keep each field concise. Optional
fields should be filled only when they are visually important.

Field guidance:
- environment_type: short category, e.g. luxury hotel room, modern coffee shop,
  classroom, office, convenience store.
- scene_location: indoor or outdoor.
- has_windows: for indoor scenes, yes if exterior windows are visible; no if
  no exterior windows are visible; unclear if uncertain. For outdoor scenes,
  use not_applicable. Glass partitions, mirrors, display cases, and interior
  glass walls do not count as exterior windows.
- style_description: environment design language and mood, not rendering style.
- required_items: 3-8 important furniture and objects that should appear.
- character_safe_zone: where a VN character could be composited without hiding
  important objects.
- identity_anchors: 3-6 details that make this location recognizable.
- avoid: people, readable text, logos, watermarks, signatures, UI overlays,
  clutter, or objects that do not fit.
- lighting: optional important light sources and mood.
- color_palette: optional key colors as short phrases.
- materials: optional visible materials such as wood, marble, fabric, metal,
  glass.
- layout_summary: optional note on where main objects are placed.
"""


BASE_PROMPT_COMPACT_TEMPLATE = """Use the attached reference image as the visual style and environment reference.
Generate an extended {environment_type} background for a visual novel.

Environment brief:
- environment_type: {environment_type}
- scene_location: {scene_location}
- has_windows: {has_windows}
- style_description: {style_description}
- required_items: {required_items}
- avoid: {avoid}

{variant_block}
Create a wider, cleaner, reusable VN background plate with the same environment
identity and design language. Use the reference image for lighting, palette,
materials, and spatial feeling without copying the exact crop. Expand the
environment logically, preserve the recognizable location identity from the
reference image, and leave clean foreground or side space where a VN character
could be placed later.

Ignore and remove any watermark, signature, stock-photo mark, border, caption,
or UI overlay from the reference image. No people, no characters, no staff, no
readable text, no logos. Wide {aspect_ratio}, eye-level camera, natural
perspective, {rendering_style}.
"""

BASE_PROMPT_FULL_TEMPLATE = """Use the attached reference image as the visual style and environment reference.
Generate an extended {environment_type} background for a visual novel.

Environment brief:
- environment_type: {environment_type}
- scene_location: {scene_location}
- has_windows: {has_windows}
- style_description: {style_description}
- lighting: {lighting}
- color_palette: {color_palette}
- materials: {materials}
- required_items: {required_items}
- layout_summary: {layout_summary}
- character_safe_zone: {character_safe_zone}
- identity_anchors: {identity_anchors}
- avoid: {avoid}

{variant_block}
Create a wider, cleaner, reusable VN background plate that keeps the same
environment identity, design language, lighting, color palette, materials, and
important items. The output does not need to copy the exact crop or aspect ratio
of the input image. It should expand the environment logically and provide a
clear character-safe area.

Ignore and remove any watermark, signature, stock-photo mark, border, caption,
or UI overlay from the reference image. No people, no characters, no staff, no
readable text, no logos. Wide {aspect_ratio}, eye-level camera, natural
perspective, {rendering_style}.
"""

OUTDOOR_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "morning",
        "label": "Morning",
        "instruction": (
            "Generate the same outdoor environment in the morning. Use fresh "
            "early daylight, gentle shadows, and a clean morning atmosphere."
        ),
    },
    {
        "id": "afternoon",
        "label": "Afternoon",
        "instruction": (
            "Generate the same outdoor environment in the afternoon. Use clear "
            "daylight, stronger natural illumination, and stable midday/afternoon "
            "shadows."
        ),
    },
    {
        "id": "evening_lights_on",
        "label": "Evening Lights On",
        "instruction": (
            "Generate the same outdoor environment in the evening or blue hour. "
            "Turn on practical lights such as street lamps, building lights, "
            "garden lights, and signage glow where appropriate, while keeping "
            "the same location identity."
        ),
    },
)

INDOOR_WITH_WINDOWS_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "daytime_curtains_open",
        "label": "Daytime Curtains Open",
        "instruction": (
            "Generate the same indoor environment during daytime with curtains "
            "or blinds open. Let natural daylight enter through the windows and "
            "keep the room clearly recognizable."
        ),
    },
    {
        "id": "night_curtains_open_lights_on",
        "label": "Night Curtains Open Lights On",
        "instruction": (
            "Generate the same indoor environment at night with curtains or "
            "blinds open. The exterior beyond the windows should look dark or "
            "nighttime, and the room's interior lights should be on."
        ),
    },
    {
        "id": "curtains_closed_indoor_light",
        "label": "Curtains Closed Indoor Light",
        "instruction": (
            "Generate the same indoor environment with curtains or blinds closed "
            "so the exterior time of day is not visible. Use only interior "
            "artificial lighting, like a completely indoor-light version."
        ),
    },
)

INDOOR_NO_WINDOWS_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "id": "bright_indoor_light",
        "label": "Bright Indoor Light",
        "instruction": (
            "Generate the same windowless indoor environment with bright, clean "
            "artificial lighting. Do not imply a visible time of day."
        ),
    },
    {
        "id": "warm_indoor_light",
        "label": "Warm Indoor Light",
        "instruction": (
            "Generate the same windowless indoor environment with warm practical "
            "interior lighting. Do not add exterior daylight or windows."
        ),
    },
    {
        "id": "dim_indoor_light",
        "label": "Dim Indoor Light",
        "instruction": (
            "Generate the same windowless indoor environment with dimmer interior "
            "lighting for an evening-like mood, but keep lights on and do not add "
            "windows or exterior time cues."
        ),
    },
)


@dataclass(frozen=True)
class Settings:
    input_dir: Path
    output_dir: Path
    api_key: str
    model: str
    api_url: str
    recursive: bool
    overwrite: bool
    dry_run: bool
    save_raw: bool
    max_images: int | None
    sleep_seconds: float
    timeout_seconds: float
    retries: int
    aspect_ratio: str
    prompt_detail: str
    rendering_style: str
    extensions: tuple[str, ...]


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description="Extract structured VN environment briefs from image references with Gemini.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing reference images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/environment_briefs"),
        help="Directory for JSON briefs, prompts, and manifest. Default: output/environment_briefs",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        help="Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model. Default: {DEFAULT_MODEL}")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"Gemini API base URL. Default: {DEFAULT_API_URL}")
    parser.add_argument("--recursive", action="store_true", help="Scan input_dir recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess images with existing JSON outputs.")
    parser.add_argument("--dry-run", action="store_true", help="List work without calling Gemini.")
    parser.add_argument("--save-raw", action="store_true", help="Save raw Gemini responses next to outputs.")
    parser.add_argument("--max-images", type=int, default=None, help="Process at most N images.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between API calls.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="HTTP request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per image after transient failures.")
    parser.add_argument("--aspect-ratio", default="16:9", help="Aspect ratio phrase for rendered prompt.")
    parser.add_argument(
        "--rendering-style",
        default=DEFAULT_RENDERING_STYLE,
        help="Fixed target rendering style inserted into every generated prompt.",
    )
    parser.add_argument(
        "--prompt-detail",
        choices=("compact", "full"),
        default="compact",
        help="How much of the extracted brief to include in the rendered prompt. Default: compact",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions. Default: .png,.jpg,.jpeg,.webp",
    )

    args = parser.parse_args()
    extensions = tuple(
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.extensions.split(",")
        if ext.strip()
    )

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY/GOOGLE_API_KEY or pass --api-key.")

    return Settings(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        api_key=args.api_key or "",
        model=args.model,
        api_url=args.api_url.rstrip("/"),
        recursive=args.recursive,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        save_raw=args.save_raw,
        max_images=args.max_images,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        aspect_ratio=args.aspect_ratio,
        prompt_detail=args.prompt_detail,
        rendering_style=args.rendering_style,
        extensions=extensions,
    )


def iter_images(input_dir: Path, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def output_paths(settings: Settings, image_path: Path) -> tuple[Path, Path, Path]:
    relative = image_path.relative_to(settings.input_dir)
    stem = relative.with_suffix("")
    json_path = settings.output_dir / "briefs" / stem.with_suffix(".environment.json")
    prompt_path = settings.output_dir / "prompts" / stem.with_suffix(".base_prompt.txt")
    raw_path = settings.output_dir / "raw" / stem.with_suffix(".gemini_response.json")
    return json_path, prompt_path, raw_path


def variant_prompt_path(settings: Settings, image_path: Path, variant_id: str) -> Path:
    relative = image_path.relative_to(settings.input_dir)
    stem = relative.with_suffix("")
    return settings.output_dir / "prompts" / stem.with_suffix(f".{variant_id}.base_prompt.txt")


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
    raise ValueError(f"Unsupported image MIME type for {image_path}")


def gemini_payload(image_path: Path) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": image_mime_type(image_path),
                            "data": image_b64,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": ENVIRONMENT_BRIEF_SCHEMA,
        },
    }


def call_gemini(settings: Settings, image_path: Path) -> dict[str, Any]:
    url = f"{settings.api_url}/models/{settings.model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.api_key,
    }
    body = json.dumps(gemini_payload(image_path)).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(1, settings.retries + 1):
        try:
            request = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {exc.code}: {error_text[:1000]}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {error_text[:1000]}")
        except Exception as exc:  # noqa: BLE001 - CLI should report the final error.
            last_error = exc
        if attempt >= settings.retries:
            break
        time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(f"Gemini request failed for {image_path}: {last_error}") from last_error


def extract_text(response_json: dict[str, Any]) -> str:
    try:
        parts = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Gemini response shape: {json.dumps(response_json)[:1000]}") from exc

    text_chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_chunks.append(part["text"])

    text = "\n".join(text_chunks).strip()
    if not text:
        raise ValueError(f"Gemini response did not contain text: {json.dumps(response_json)[:1000]}")
    return text


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Gemini JSON output must be an object.")
    return normalize_brief(data)


def normalize_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        pieces = [piece.strip(" -") for piece in value.replace("\n", ",").split(",")]
        return [piece for piece in pieces if piece]
    if value is None:
        return []
    return [str(value).strip()]


def normalize_scene_location(value: Any) -> str:
    cleaned = str(value).strip().lower().replace("-", "_")
    if any(token in cleaned for token in ("outdoor", "outside", "exterior")):
        return "outdoor"
    if any(token in cleaned for token in ("indoor", "inside", "interior")):
        return "indoor"
    return "indoor"


def normalize_has_windows(value: Any, scene_location: str) -> str:
    if scene_location == "outdoor":
        return "not_applicable"

    cleaned = str(value).strip().lower().replace("-", "_")
    if cleaned in {"yes", "true", "y", "has_windows", "has window", "has windows"}:
        return "yes"
    if cleaned in {"no", "false", "n", "none", "no_windows", "windowless"}:
        return "no"
    if cleaned in {"unclear", "unknown", "maybe"}:
        return "unclear"
    if "windowless" in cleaned or "no window" in cleaned or "without window" in cleaned:
        return "no"
    if "window" in cleaned:
        return "yes"
    return "unclear"


def normalize_brief(data: dict[str, Any]) -> dict[str, Any]:
    brief: dict[str, Any] = {}
    for field in BRIEF_FIELDS:
        value = data.get(field, "")
        if field in LIST_FIELDS:
            brief[field] = normalize_list(value)
        else:
            brief[field] = str(value).strip()
    brief["scene_location"] = normalize_scene_location(brief.get("scene_location", ""))
    brief["has_windows"] = normalize_has_windows(
        brief.get("has_windows", ""),
        brief["scene_location"],
    )
    return brief


def fmt_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def fmt_prompt_value(field: str, value: Any) -> str:
    if isinstance(value, list):
        limit = PROMPT_LIST_LIMITS.get(field)
        items = value[:limit] if limit else value
        return ", ".join(str(item) for item in items)
    return str(value)


def render_base_prompt(
    brief: dict[str, Any],
    aspect_ratio: str,
    prompt_detail: str,
    rendering_style: str,
    variant_instruction: str | None = None,
) -> str:
    values = {field: fmt_prompt_value(field, brief.get(field, "")) for field in BRIEF_FIELDS}
    values["aspect_ratio"] = aspect_ratio
    values["rendering_style"] = rendering_style
    values["variant_block"] = (
        f"Lighting/time condition:\n{variant_instruction}\n" if variant_instruction else ""
    )
    template = BASE_PROMPT_FULL_TEMPLATE if prompt_detail == "full" else BASE_PROMPT_COMPACT_TEMPLATE
    return template.format(**values).strip() + "\n"


def get_variant_specs(brief: dict[str, Any]) -> tuple[dict[str, str], ...]:
    scene_location = normalize_scene_location(brief.get("scene_location", ""))
    has_windows = normalize_has_windows(brief.get("has_windows", ""), scene_location)

    if scene_location == "outdoor":
        return OUTDOOR_VARIANTS
    if has_windows == "yes":
        return INDOOR_WITH_WINDOWS_VARIANTS
    return INDOOR_NO_WINDOWS_VARIANTS


def render_variant_prompts(
    brief: dict[str, Any],
    aspect_ratio: str,
    prompt_detail: str,
    rendering_style: str,
) -> dict[str, dict[str, str]]:
    variants: dict[str, dict[str, str]] = {}
    for spec in get_variant_specs(brief):
        prompt = render_base_prompt(
            brief,
            aspect_ratio,
            prompt_detail,
            rendering_style,
            variant_instruction=spec["instruction"],
        )
        variants[spec["id"]] = {
            "label": spec["label"],
            "instruction": spec["instruction"],
            "prompt": prompt,
        }
    return variants


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_manifest(settings: Settings, row: dict[str, Any]) -> None:
    manifest_path = settings.output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_image(settings: Settings, image_path: Path) -> bool:
    json_path, prompt_path, raw_path = output_paths(settings, image_path)
    if json_path.exists() and prompt_path.exists() and not settings.overwrite:
        print(f"skip existing: {image_path}")
        return False

    if settings.dry_run:
        print(f"would process: {image_path}")
        return False

    print(f"process: {image_path}")
    raw_response = call_gemini(settings, image_path)
    brief = parse_json_text(extract_text(raw_response))
    base_prompt = render_base_prompt(
        brief,
        settings.aspect_ratio,
        settings.prompt_detail,
        settings.rendering_style,
    )
    variant_prompts = render_variant_prompts(
        brief,
        settings.aspect_ratio,
        settings.prompt_detail,
        settings.rendering_style,
    )

    package = {
        "source_image": str(image_path),
        "model": settings.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_detail": settings.prompt_detail,
        "fixed_rendering_style": settings.rendering_style,
        "brief": brief,
        "base_prompt": base_prompt,
        "variant_prompts": variant_prompts,
    }

    write_json(json_path, package)
    write_text(prompt_path, base_prompt)
    variant_prompt_paths = {}
    for variant_id, variant_data in variant_prompts.items():
        path = variant_prompt_path(settings, image_path, variant_id)
        write_text(path, variant_data["prompt"])
        variant_prompt_paths[variant_id] = str(path)
    if settings.save_raw:
        write_json(raw_path, raw_response)

    append_manifest(
        settings,
        {
            "source_image": str(image_path),
            "brief_json": str(json_path),
            "base_prompt": str(prompt_path),
            "variant_prompts": variant_prompt_paths,
            "environment_type": brief.get("environment_type", ""),
            "scene_location": brief.get("scene_location", ""),
            "has_windows": brief.get("has_windows", ""),
            "fixed_rendering_style": settings.rendering_style,
            "created_at": package["created_at"],
        },
    )
    return True


def main() -> int:
    settings = parse_args()
    images = iter_images(settings.input_dir, settings.recursive, settings.extensions)
    if settings.max_images is not None:
        images = images[: settings.max_images]

    print(f"input: {settings.input_dir}")
    print(f"output: {settings.output_dir}")
    print(f"model: {settings.model}")
    print(f"images: {len(images)}")

    processed = 0
    failed = 0
    for image_path in images:
        try:
            if process_image(settings, image_path):
                processed += 1
                if settings.sleep_seconds > 0:
                    time.sleep(settings.sleep_seconds)
        except Exception as exc:  # noqa: BLE001 - continue batch after failures.
            failed += 1
            print(f"ERROR: {image_path}: {exc}", file=sys.stderr)

    print(f"done: processed={processed}, skipped_or_dry={len(images) - processed - failed}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
