"""Grok (xAI) chat client — OpenAI-compatible /chat/completions, Bearer auth.

Grok is this repo's default hosted LLM for text/vision work.
"""

from __future__ import annotations

from typing import Any

from lib.llm_client.base import APIError, BaseAPIClient, OnRetry


class GrokClient(BaseAPIClient):
    SERVICE_NAME = "Grok (xAI)"
    ENV_KEYS = ("XAI_API_KEY",)
    DEFAULT_BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODEL = "grok-4.3"

    def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        retries: int = 1,
        timeout: float | None = None,
        on_retry: OnRetry | None = None,
    ) -> dict[str, Any]:
        """POST /chat/completions and return the full response dict."""
        return self.post_json(
            "/chat/completions", payload, retries=retries, timeout=timeout, on_retry=on_retry,
        )

    def chat_text(
        self,
        payload: dict[str, Any],
        *,
        retries: int = 1,
        timeout: float | None = None,
        on_retry: OnRetry | None = None,
    ) -> str:
        """POST /chat/completions and return the assistant message content string."""
        response = self.chat_completions(payload, retries=retries, timeout=timeout, on_retry=on_retry)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"unexpected chat response shape: {exc!r}") from exc
        if not isinstance(content, str):
            raise APIError("unexpected chat response shape: message content is not a string")
        return content
