"""OpenAI REST client (images, files, batches) — Bearer auth.

Only the endpoints this repo needs are wrapped; the generic
get_json/post_json/post_multipart verbs from BaseAPIClient cover the rest.
"""

from __future__ import annotations

from typing import Any

from lib.llm_client.base import APIError, BaseAPIClient

# Backward-compatible alias: earlier copies of this client raised OpenAIError.
OpenAIError = APIError


class OpenAIClient(BaseAPIClient):
    SERVICE_NAME = "OpenAI"
    ENV_KEYS = ("OPENAI_API_KEY",)
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT: float = 600

    def upload_file(self, filename: str, data: bytes, purpose: str) -> dict[str, Any]:
        return self.post_multipart(
            "/files",
            fields=[("purpose", purpose)],
            files=[("file", filename, data, "application/octet-stream")],
        )

    def file_content(self, file_id: str) -> bytes:
        return self.request_bytes(f"/files/{file_id}/content")

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata
        return self.post_json("/batches", payload)

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        return self.get_json(f"/batches/{batch_id}")
