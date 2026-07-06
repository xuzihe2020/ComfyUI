"""Load .env credentials without overwriting real environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from lib.paths import REPO_ROOT

DEFAULT_ENV_FILE = REPO_ROOT / ".env"
_PLACEHOLDER_MARKERS = ("replace_with", "dummy")


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from .env without overwriting real env vars."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def env_api_key(*keys: str) -> str:
    """Return the first non-placeholder value among the given env var names."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value and not any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
            return value
    return ""
