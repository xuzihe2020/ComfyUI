"""Gemini (Google AI) client — generateContent with x-goog-api-key auth."""

from __future__ import annotations

import json
from typing import Any

from lib.llm_client.base import BaseAPIClient, OnRetry


class GeminiClient(BaseAPIClient):
    SERVICE_NAME = "Gemini"
    ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key}

    def generate_content(
        self,
        model: str,
        payload: dict[str, Any],
        *,
        retries: int = 3,
        backoff_cap: float = 8,
        timeout: float | None = None,
        on_retry: OnRetry | None = None,
    ) -> dict[str, Any]:
        """POST models/{model}:generateContent and return the full response dict."""
        return self.post_json(
            f"/models/{model}:generateContent",
            payload,
            retries=retries,
            backoff_cap=backoff_cap,
            timeout=timeout,
            on_retry=on_retry,
        )

    @staticmethod
    def extract_text(response_json: dict[str, Any]) -> str:
        """Join the text parts of the first candidate; raise on unexpected shape."""
        try:
            parts = response_json["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Unexpected Gemini response shape: {json.dumps(response_json)[:1000]}"
            ) from exc

        text_chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_chunks.append(part["text"])

        text = "\n".join(text_chunks).strip()
        if not text:
            raise ValueError(
                f"Gemini response did not contain text: {json.dumps(response_json)[:1000]}"
            )
        return text
