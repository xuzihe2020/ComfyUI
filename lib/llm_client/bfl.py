"""BFL (Black Forest Labs) client — x-key auth, asynchronous generation.

POST the model endpoint, then poll the returned polling_url until the task is
Ready and download the signed result URL (valid ~10 minutes). Keys come from
https://dashboard.bfl.ai.
"""

from __future__ import annotations

import json
import time
from typing import Any

from lib.llm_client.base import APIError, BaseAPIClient

TERMINAL_FAILURE_STATUSES = {"Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"}


class BFLError(APIError):
    pass


class BFLClient(BaseAPIClient):
    SERVICE_NAME = "BFL"
    ENV_KEYS = ("FLUX_API_KEY", "BFL_API_KEY")
    DEFAULT_BASE_URL = "https://api.bfl.ai"

    def auth_headers(self) -> dict[str, str]:
        return {"x-key": self.api_key}

    def download(self, url: str) -> bytes:
        """Fetch a signed result URL (no auth header needed)."""
        return self.request_bytes(url, with_auth=False)

    def poll_until_ready(self, polling_url: str, timeout_s: int, interval_s: float = 2.0) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while True:
            body = self.get_json(polling_url)
            status = body.get("status")
            if status == "Ready":
                return body
            if status in TERMINAL_FAILURE_STATUSES:
                raise BFLError(f"BFL task ended as {status}: {json.dumps(body)[:500]}")
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out after {timeout_s}s waiting for {polling_url} (last status: {status})")
            time.sleep(interval_s)
