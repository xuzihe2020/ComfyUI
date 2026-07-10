"""Filesystem layer for the LoRA caption tool.

Image discovery, data-URI encoding, prompt loading, and output writing.

Data-flow rule: the per-image JSON under the captures directory is the source
of truth (raw captions with the {TRIGGER} placeholder); the sibling .txt next
to each training image is a derived artifact regenerated from the JSON on
every run. Edit captions in the JSON, never in the .txt.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
# xAI documents jpg/jpeg/png as the reliably-supported input types and a 20 MiB cap.
XAI_SAFE_MIME = {"image/jpeg", "image/png"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def output_base(image: Path, input_dir: Path) -> str:
    """A collision-safe base name preserving subdirectory structure via '__'."""
    rel = image.relative_to(input_dir)
    parts = list(rel.with_suffix("").parts)
    return "__".join(parts)


def caption_txt_path(image: Path) -> Path:
    """Sibling training caption: my_image.jpeg -> my_image.txt, same folder."""
    return image.with_suffix(".txt")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_if_changed(path: Path, text: str) -> bool:
    """Write text; return True when the file was created or its content changed."""
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True
