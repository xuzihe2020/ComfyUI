"""Discovery of video files to sample."""

from __future__ import annotations

from pathlib import Path

# Spec targets webm and mp4; the rest are common containers PyAV handles too.
DEFAULT_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi")


def discover_videos(
    path: str | Path,
    recursive: bool = False,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Return a sorted list of video files for ``path``.

    ``path`` may be a single video file or a directory. Directories are scanned
    (optionally recursively) for files whose suffix is in ``extensions``.
    """
    root = Path(path).expanduser()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}

    if not root.exists():
        raise FileNotFoundError(f"input path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() not in exts:
            raise ValueError(
                f"{root} is not a recognized video file (extensions: {sorted(exts)})"
            )
        return [root]

    pattern = "**/*" if recursive else "*"
    videos = sorted(
        p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts
    )
    return videos
