"""Aspect-ratio-preserving resize.

Three ways to specify the target, none of which stretch, deform, or crop:

* ``scale``           - uniform percentage (0.8 -> 80% of each side).
* ``width`` OR ``height`` - the other dimension is derived to keep the aspect ratio.
* ``width`` AND ``height`` - allowed only if they already form the original aspect
  ratio (i.e. a pure uniform scale, within 1px of integer rounding); otherwise
  honoring both would require stretching or cropping, so it raises ImageOpError.
"""

from __future__ import annotations

from lib.errors import ImageOpError
from lib.imaging import RESAMPLE


def validate_resize_spec(scale, width, height) -> None:
    """Static (image-independent) validation of the resize arguments."""
    if scale is not None:
        if width is not None or height is not None:
            raise ImageOpError("--scale cannot be combined with --width/--height")
        if scale <= 0:
            raise ImageOpError(f"--scale must be > 0 (got {scale})")
        return
    if width is None and height is None:
        raise ImageOpError("resize requires --scale, --width, and/or --height")
    if width is not None and width <= 0:
        raise ImageOpError(f"--width must be > 0 (got {width})")
    if height is not None and height <= 0:
        raise ImageOpError(f"--height must be > 0 (got {height})")


def compute_resize_size(orig_w, orig_h, *, scale=None, width=None, height=None):
    """Resolve the target (width, height) for one image."""
    if scale is not None:
        return max(1, round(orig_w * scale)), max(1, round(orig_h * scale))

    if width is not None and height is not None:
        # Both given: must be a pure uniform scale of the original, allowing 1px
        # of integer-rounding slack from either dimension.
        matches = (
            round(orig_h * width / orig_w) == height
            or round(orig_w * height / orig_h) == width
        )
        if not matches:
            raise ImageOpError(
                f"width={width} height={height} (ratio {width / height:.4f}) does not "
                f"match the original {orig_w}x{orig_h} aspect ratio "
                f"({orig_w / orig_h:.4f}); resize never stretches or crops. "
                f"For width={width}, height should be ~{round(orig_h * width / orig_w)}."
            )
        return width, height

    if width is not None:
        return width, max(1, round(orig_h * width / orig_w))
    return max(1, round(orig_w * height / orig_h)), height


def resize_image(img, *, scale=None, width=None, height=None):
    target = compute_resize_size(
        img.width, img.height, scale=scale, width=width, height=height
    )
    if target == (img.width, img.height):
        return img
    return img.resize(target, RESAMPLE)
