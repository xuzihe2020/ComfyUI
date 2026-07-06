"""Base class for the hosted-API clients in lib.llm_client.

BaseAPIClient owns everything the concrete clients share:

- API-key resolution from the environment / repo-root .env (lib.envfile)
  through `from_env()`, with a clear error naming the expected env vars;
- auth headers (Bearer by default, overridden by services with custom key
  headers) plus optional caller-supplied extra headers;
- a urllib-only transport (no SDK dependencies, by repo convention) with
  retry/backoff on 429/5xx and network errors, HTTP error detail extraction,
  and multipart/form-data encoding.

Subclasses declare SERVICE_NAME / ENV_KEYS / DEFAULT_BASE_URL (and optionally
DEFAULT_TIMEOUT / DEFAULT_MODEL), override auth_headers() when the service
does not use Bearer auth, and add endpoint methods on top of the generic
get_json / post_json / post_multipart verbs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from lib.envfile import env_api_key, load_env_file

# Retry only on rate limits and server errors; other 4xx raise immediately.
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}

# on_retry(attempt, delay_seconds, error) — called before each backoff sleep so
# callers can keep their own logging style.
OnRetry = Callable[[int, float, Exception], None]


class APIError(RuntimeError):
    """HTTP API failure carrying the status code and response body detail."""

    def __init__(self, message: str, status: int | None = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


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


class BaseAPIClient:
    SERVICE_NAME = "API"
    ENV_KEYS: tuple[str, ...] = ()
    DEFAULT_BASE_URL = ""
    DEFAULT_TIMEOUT: float = 120

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise APIError(
                f"missing {self.SERVICE_NAME} API key; set "
                f"{' or '.join(self.ENV_KEYS)} in the environment or the repo .env"
            )
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = self.DEFAULT_TIMEOUT if timeout is None else timeout
        self.extra_headers = dict(extra_headers or {})

    @classmethod
    def from_env(
        cls,
        api_key: str = "",
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
        env_keys: tuple[str, ...] | None = None,
    ):
        """Build a client with the key from the environment / repo-root .env."""
        load_env_file()
        key = api_key or env_api_key(*(env_keys or cls.ENV_KEYS))
        return cls(key, base_url=base_url, timeout=timeout, extra_headers=extra_headers)

    # ------------------------------------------------------------------ #
    # Headers
    # ------------------------------------------------------------------ #

    def auth_headers(self) -> dict[str, str]:
        """Bearer auth by default; override for services with custom key headers."""
        return {"Authorization": f"Bearer {self.api_key}"}

    def _headers(self, content_type: str | None = None, with_auth: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if with_auth:
            headers.update(self.auth_headers())
            headers.update(self.extra_headers)
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return self.base_url + path_or_url

    def request_bytes(
        self,
        path_or_url: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
        content_type: str | None = None,
        with_auth: bool = True,
        timeout: float | None = None,
        retries: int = 1,
        backoff_cap: float = 30,
        transient_codes: set[int] | None = None,
        on_retry: OnRetry | None = None,
    ) -> bytes:
        """urllib request with retry/backoff. `retries` is the TOTAL attempt count."""
        url = self._url(path_or_url)
        headers = self._headers(content_type, with_auth=with_auth)
        request_timeout = self.timeout if timeout is None else timeout
        retry_codes = TRANSIENT_HTTP_CODES if transient_codes is None else transient_codes

        attempts = max(1, retries)
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_err = APIError(
                    f"HTTP {exc.code} on {method} {url.split('?')[0]}: {detail[:2000]}",
                    status=exc.code,
                    detail=detail,
                )
                if exc.code not in retry_codes:
                    raise last_err from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc

            if attempt < attempts:
                delay = float(min(2 ** attempt, backoff_cap))
                if on_retry is not None:
                    on_retry(attempt, delay, last_err)
                time.sleep(delay)

        if isinstance(last_err, APIError):
            raise last_err
        raise APIError(
            f"{method} {url.split('?')[0]} failed after {attempts} attempt(s): {last_err}"
        ) from last_err

    def request_json(self, path_or_url: str, **kwargs: Any) -> Any:
        return json.loads(self.request_bytes(path_or_url, **kwargs).decode("utf-8"))

    # ------------------------------------------------------------------ #
    # Generic REST verbs shared by all clients
    # ------------------------------------------------------------------ #

    def get_json(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.request_json(path, **kwargs)

    def post_json(self, path: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.request_json(
            path,
            method="POST",
            body=json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            **kwargs,
        )

    def post_multipart(
        self,
        path: str,
        fields: list[tuple[str, str]],
        files: list[tuple[str, str, bytes, str]],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body, content_type = encode_multipart(fields, files)
        return self.request_json(
            path, method="POST", body=body, content_type=content_type, timeout=timeout, **kwargs,
        )
