"""Discovery of image files to process."""

from __future__ import annotations

from pathlib import Path

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def discover_images(
    path: str | Path,
    recursive: bool = False,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Return a sorted list of image files for ``path`` (a file or directory)."""
    root = Path(path).expanduser()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    if not root.exists():
        raise FileNotFoundError(f"input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in exts:
            raise ValueError(
                f"{root} is not a recognized image (extensions: {sorted(exts)})"
            )
        return [root]

    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts
    )
