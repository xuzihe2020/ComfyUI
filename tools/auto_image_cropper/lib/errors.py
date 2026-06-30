"""Shared exception types."""

from __future__ import annotations


class ImageOpError(ValueError):
    """Raised when a resize/crop request is invalid for a given image."""
