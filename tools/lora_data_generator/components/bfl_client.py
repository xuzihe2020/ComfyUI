"""Minimal stdlib client for the official BFL (Black Forest Labs) REST API.

Auth is an x-key header with a key from https://dashboard.bfl.ai. Generation is
asynchronous: POST the model endpoint, then poll the returned polling_url until
the task is Ready and download the signed result URL (valid ~10 minutes).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

TERMINAL_FAILURE_STATUSES = {"Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"}


class BFLError(RuntimeError):
    pass


class BFLClient:
    def __init__(self, api_key: str, base_url: str = "https://api.bfl.ai", timeout: int = 120) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, url: str, body: bytes | None = None, with_key: bool = True) -> bytes:
        headers: dict[str, str] = {}
        if with_key:
            headers["x-key"] = self.api_key
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BFLError(f"BFL API error {exc.code} on {method} {url.split('?')[0]}: {detail}") from exc

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return json.loads(self._request("POST", self.base_url + path, body=body).decode("utf-8"))

    def get_json(self, url: str) -> dict[str, Any]:
        if not url.startswith("http"):
            url = self.base_url + url
        return json.loads(self._request("GET", url).decode("utf-8"))

    def download(self, url: str) -> bytes:
        """Fetch a signed result URL (no auth header needed)."""
        return self._request("GET", url, with_key=False)

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
