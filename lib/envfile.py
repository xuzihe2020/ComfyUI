"""Compatibility environment API backed by the canonical shared parser."""

from __future__ import annotations

from pathlib import Path

from aigc_shared.config import load_env_file as _load_shared_env_file
from aigc_shared.config import load_settings_env
from aigc_shared.envfile import env_api_key

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load a ComfyUI environment file with the legacy quote behavior."""

    _load_shared_env_file(path, literal_quotes=False)


def env_value(*keys: str) -> str:
    """Resolve process, ComfyUI project, then shared user settings."""

    load_settings_env(project_env=DEFAULT_ENV_FILE)
    return env_api_key(*keys)
