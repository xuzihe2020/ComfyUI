"""Center crop, specified either by target size or by symmetric margins.

Both inputs produce the same centered region; they differ only in how you express
it. For a 1024x1536 image:

* ``size   = (1000, 1500)`` -> trims (1024-1000)/2 = 12 px off left & right,
                               (1536-1500)/2 = 18 px off top & bottom.
* ``margin = (12, 18)``     -> trims width-margin 12 px off left & right,
                               height-margin 18 px off top & bottom.

Both yield the identical centered 1000x1500 image. Crop only ever keeps the
center region; the two input modes exist for different conveniences.
"""

from __future__ import annotations

from lib.errors import ImageOpError


def compute_crop_box(orig_w, orig_h, *, size=None, margin=None):
    """Return the (left, top, right, bottom) box for a centered crop."""
    if (size is None) == (margin is None):
        raise ImageOpError("crop requires exactly one of --size or --margin")

    if size is not None:
        w, h = size
        if w <= 0 or h <= 0:
            raise ImageOpError(f"crop --size must be positive (got {w}x{h})")
        if w > orig_w or h > orig_h:
            raise ImageOpError(
                f"crop size {w}x{h} is larger than the image {orig_w}x{orig_h}; "
                f"a center crop cannot enlarge."
            )
        left = (orig_w - w) // 2
        top = (orig_h - h) // 2
        return left, top, left + w, top + h

    mw, mh = margin
    if mw < 0 or mh < 0:
        raise ImageOpError(f"crop --margin must be non-negative (got {mw},{mh})")
    if 2 * mw >= orig_w or 2 * mh >= orig_h:
        raise ImageOpError(
            f"crop margins ({mw},{mh}) would remove the entire {orig_w}x{orig_h} image."
        )
    return mw, mh, orig_w - mw, orig_h - mh


def crop_image(img, *, size=None, margin=None):
    box = compute_crop_box(img.width, img.height, size=size, margin=margin)
    return img.crop(box)
