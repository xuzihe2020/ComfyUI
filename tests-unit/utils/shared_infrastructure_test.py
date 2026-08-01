"""Regression tests for ComfyUI's ``aigc_shared`` compatibility boundary."""

from __future__ import annotations

import lib.envfile as legacy_envfile
from aigc_shared.llm_client import (
    APIError as SharedAPIError,
    BFLClient as SharedBFLClient,
    BaseAPIClient as SharedBaseAPIClient,
    GeminiClient as SharedGeminiClient,
    GrokClient as SharedGrokClient,
    OpenAIClient as SharedOpenAIClient,
)
from lib.llm_client import (
    APIError,
    BFLClient,
    BaseAPIClient,
    GeminiClient,
    GrokClient,
    OpenAIClient,
)


def test_legacy_client_imports_are_canonical_shared_classes() -> None:
    assert APIError is SharedAPIError
    assert BaseAPIClient is SharedBaseAPIClient
    assert BFLClient is SharedBFLClient
    assert GeminiClient is SharedGeminiClient
    assert GrokClient is SharedGrokClient
    assert OpenAIClient is SharedOpenAIClient


def test_comfy_env_wrapper_loads_project_file_without_overwriting_process_env(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'COMFY_SHARED_FILE_VALUE="decoded value"\nCOMFY_SHARED_PROCESS_VALUE=file\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy_envfile, "DEFAULT_ENV_FILE", env_file)
    monkeypatch.delenv("COMFY_SHARED_FILE_VALUE", raising=False)
    monkeypatch.setenv("COMFY_SHARED_PROCESS_VALUE", "process")

    assert legacy_envfile.env_value("COMFY_SHARED_FILE_VALUE") == "decoded value"
    assert legacy_envfile.env_value("COMFY_SHARED_PROCESS_VALUE") == "process"
