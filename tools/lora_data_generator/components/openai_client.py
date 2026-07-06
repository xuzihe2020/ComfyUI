"""Minimal stdlib client for the OpenAI REST API (images, files, batches).

Kept dependency-free on purpose so the tool runs in the ComfyUI venv without
installing the openai SDK. Only the endpoints this pipeline needs are wrapped.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any


class OpenAIError(RuntimeError):
    pass


def encode_multipart(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    """Encode form fields and (name, filename, data, content_type) files."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields:
        parts += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    for name, filename, data, content_type in files:
        parts += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode(),
            f"Content-Type: {content_type}".encode(),
            b"",
            data,
        ]
    parts += [f"--{boundary}--".encode(), b""]
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: int = 600) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
        timeout: int | None = None,
    ) -> bytes:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAIError(f"OpenAI API error {exc.code} on {method} {path}: {detail}") from exc

    def get_json(self, path: str) -> dict[str, Any]:
        return json.loads(self._request("GET", path).decode("utf-8"))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return json.loads(self._request("POST", path, body=body, content_type="application/json").decode("utf-8"))

    def post_multipart(
        self,
        path: str,
        fields: list[tuple[str, str]],
        files: list[tuple[str, str, bytes, str]],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        body, content_type = encode_multipart(fields, files)
        return json.loads(
            self._request("POST", path, body=body, content_type=content_type, timeout=timeout).decode("utf-8")
        )

    def upload_file(self, filename: str, data: bytes, purpose: str) -> dict[str, Any]:
        return self.post_multipart(
            "/files",
            fields=[("purpose", purpose)],
            files=[("file", filename, data, "application/octet-stream")],
        )

    def file_content(self, file_id: str) -> bytes:
        return self._request("GET", f"/files/{file_id}/content")

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
