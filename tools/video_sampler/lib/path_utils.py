"""Filesystem-safe path helpers for sampler outputs."""

from __future__ import annotations

import re


_WINDOWS_RESERVED_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def safe_output_stem(stem: str) -> str:
    """Return a Windows-safe output stem derived from a video filename stem."""
    value = _WINDOWS_RESERVED_CHARS.sub("_", stem)
    value = _WHITESPACE.sub(" ", value).strip(" .")
    return value or "video"
