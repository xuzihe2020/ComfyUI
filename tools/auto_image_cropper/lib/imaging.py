"""Image load/save helpers (Pillow)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# High-quality resampling filter: best for downscaling, fine for upscaling.
RESAMPLE = Image.LANCZOS

_LOSSY = {".jpg", ".jpeg", ".webp"}


def load_image(path) -> Image.Image:
    img = Image.open(path)
    img.load()  # force decode now so errors surface here, not later
    return img


def save_image(img: Image.Image, out_path: Path, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    params: dict = {}
    if ext in _LOSSY:
        params["quality"] = quality
        if ext in (".jpg", ".jpeg") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")  # JPEG cannot store alpha/palette
    img.save(out_path, **params)
