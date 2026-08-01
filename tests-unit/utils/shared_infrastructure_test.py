"""Regression tests for ComfyUI's ``aigc_shared`` compatibility boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import lib.envfile as legacy_envfile
import lib.llm_client.base as legacy_base
import lib.llm_client.bfl as legacy_bfl
import lib.llm_client.gemini as legacy_gemini
import lib.llm_client.grok as legacy_grok
import lib.llm_client.openai as legacy_openai
import pytest
import aigc_shared.llm_client.base as shared_base
import aigc_shared.llm_client.bfl as shared_bfl
import aigc_shared.llm_client.gemini as shared_gemini
import aigc_shared.llm_client.grok as shared_grok
import aigc_shared.llm_client.openai as shared_openai
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
from scripts.environment.gemini_environment_briefs import (
    append_manifest as append_gemini_manifest,
    image_mime_type as gemini_image_mime_type,
    write_json as write_gemini_json,
)
from scripts.environment.openai_generate_environment_images import (
    append_manifest as append_openai_manifest,
    image_mime_type as openai_image_mime_type,
    write_json as write_openai_json,
)
from scripts.image_description.describe_images import parse_args as parse_description_args


def test_legacy_client_imports_are_canonical_shared_classes() -> None:
    assert APIError is SharedAPIError
    assert BaseAPIClient is SharedBaseAPIClient
    assert BFLClient is SharedBFLClient
    assert GeminiClient is SharedGeminiClient
    assert GrokClient is SharedGrokClient
    assert OpenAIClient is SharedOpenAIClient


def test_legacy_client_submodules_are_canonical_shared_exports() -> None:
    assert legacy_base.APIError is shared_base.APIError
    assert legacy_base.BaseAPIClient is shared_base.BaseAPIClient
    assert legacy_base.encode_multipart is shared_base.encode_multipart
    assert legacy_bfl.BFLClient is shared_bfl.BFLClient
    assert legacy_bfl.BFLError is shared_bfl.BFLError
    assert legacy_gemini.GeminiClient is shared_gemini.GeminiClient
    assert legacy_grok.GrokClient is shared_grok.GrokClient
    assert legacy_openai.OpenAIClient is shared_openai.OpenAIClient
    assert legacy_openai.OpenAIError is shared_openai.OpenAIError


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


def test_grok_script_defaults_use_canonical_xai_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("XAI_MODEL", "grok-4.3")
    monkeypatch.setenv("XAI_API_BASE_URL", "https://api.x.ai/v1")

    args = parse_description_args([str(tmp_path)])

    assert args.model == "grok-4.3"
    assert args.base_url == "https://api.x.ai/v1"


def test_environment_scripts_share_supported_image_mime_detection() -> None:
    for path, expected in (
        ("reference.JPG", "image/jpeg"),
        ("reference.png", "image/png"),
        ("reference.webp", "image/webp"),
    ):
        assert gemini_image_mime_type(Path(path)) == expected
        assert openai_image_mime_type(Path(path)) == expected


def test_environment_scripts_preserve_unsupported_image_errors() -> None:
    with pytest.raises(ValueError, match="Unsupported image MIME type"):
        gemini_image_mime_type(Path("reference.gif"))
    with pytest.raises(ValueError, match="Unsupported reference image type"):
        openai_image_mime_type(Path("reference.gif"))


def test_environment_scripts_preserve_json_and_manifest_bytes(tmp_path) -> None:
    payload = {"name": "环境", "value": 2}
    for name, writer in (
        ("gemini", write_gemini_json),
        ("openai", write_openai_json),
    ):
        output = tmp_path / name / "record.json"
        writer(output, payload)
        assert output.read_text(encoding="utf-8") == (
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
        assert not (output.parent / f".{output.name}.tmp").exists()

    gemini_settings = SimpleNamespace(output_dir=tmp_path / "gemini-output")
    openai_settings = SimpleNamespace(output_dir=tmp_path / "openai-output")
    append_gemini_manifest(gemini_settings, payload)
    append_openai_manifest(openai_settings, payload)
    expected = json.dumps(payload, ensure_ascii=False) + "\n"
    assert (gemini_settings.output_dir / "manifest.jsonl").read_text() == expected
    assert (
        openai_settings.output_dir / "generation_manifest.jsonl"
    ).read_text() == expected
