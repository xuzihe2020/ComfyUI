"""Reference image resolution and indexing shared by both pipelines.

References are ordered dressing refs first, then character refs. Their 1-based
position in that list is the index the prompt uses to address them (see
lib.prompting.reference_order_block), so both pipelines must attach images to
their respective APIs in exactly this order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_lib.jobs import list_field
from tool_lib.paths import REPO_ROOT

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_DRESSING_REFS = 2
MAX_CHARACTER_REFS = 5
MAX_TOTAL_REFS = MAX_DRESSING_REFS + MAX_CHARACTER_REFS

REF_ROLE_BY_KIND = {
    "dressing": "dressing/outfit reference",
    "character": "character face/body identity reference",
}


def resolve_image_path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Reference image path must be a non-empty string, got {value!r}")
    raw = Path(value.strip())
    candidates = [raw] if raw.is_absolute() else [base_dir / raw, REPO_ROOT / raw, raw]
    for candidate in candidates:
        if candidate.is_file():
            suffix = candidate.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension for {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(f"Reference image not found: {value}")


def reference_paths(item: dict[str, Any], base_dir: Path) -> list[tuple[str, Path]]:
    dressing = list_field(item, "dressing_reference_images", "dressing_references", "dressing_refs")
    character = list_field(item, "character_reference_images", "character_references", "character_refs")
    if len(dressing) > MAX_DRESSING_REFS:
        raise ValueError(f"Expected at most {MAX_DRESSING_REFS} dressing references, got {len(dressing)}")
    if len(character) > MAX_CHARACTER_REFS:
        raise ValueError(f"Expected at most {MAX_CHARACTER_REFS} character references, got {len(character)}")

    refs = [("dressing", resolve_image_path(path, base_dir)) for path in dressing]
    refs.extend(("character", resolve_image_path(path, base_dir)) for path in character)
    if len(refs) > MAX_TOTAL_REFS:
        raise ValueError(f"Expected at most {MAX_TOTAL_REFS} total references, got {len(refs)}")
    return refs


def ensure_extensions(refs: list[tuple[str, Path]], allowed: set[str], context: str) -> None:
    for _, path in refs:
        if path.suffix.lower() not in allowed:
            raise ValueError(
                f"{context} does not accept {path.suffix} reference images ({path}); "
                f"allowed: {', '.join(sorted(allowed))}"
            )


def ref_summary(refs: list[tuple[str, Path]]) -> str:
    dressing = sum(1 for kind, _ in refs if kind == "dressing")
    character = sum(1 for kind, _ in refs if kind == "character")
    return f"{dressing} dressing, {character} character"


def reference_log_entries(refs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "kind": kind,
            "role": REF_ROLE_BY_KIND[kind],
            "path": str(path),
            "filename": path.name,
        }
        for index, (kind, path) in enumerate(refs, start=1)
    ]
