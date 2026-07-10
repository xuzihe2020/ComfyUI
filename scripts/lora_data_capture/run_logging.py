"""Token/cost accounting for Grok traffic, with optional JSONL file logging.

Accounting (per-attempt usage, run totals) always runs. File logging is
enabled by passing a logs directory (debug mode); pass ``None`` to account
without writing anything.

When file logging is enabled, one log file per run (``run_<timestamp>.jsonl``)
is written to the logs directory (default ``<input_dir>/logs``). Every Grok
attempt is one JSON line holding:

- the raw request body sent to Grok, with image data URIs replaced by a
  length stub (a base64 image would bloat the log by megabytes per line);
- the raw API response, verbatim (message content, usage, ids);
- the token usage split into input/output and the estimated USD cost.

The file starts with a ``run_start`` line and ends with a ``summary`` line.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pricing


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-ish copy of chat messages with image data URIs omitted."""
    safe_messages = []
    for message in messages:
        safe_message = {"role": message.get("role"), "content": message.get("content")}
        content = safe_message["content"]
        if not isinstance(content, list):
            safe_messages.append(safe_message)
            continue

        safe_content = []
        for part in content:
            safe_part = dict(part)
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                safe_part["image_url"] = {
                    "url": f"<data-uri omitted, {len(url)} chars>",
                    "detail": part.get("image_url", {}).get("detail"),
                }
            safe_content.append(safe_part)
        safe_message["content"] = safe_content
        safe_messages.append(safe_message)
    return safe_messages


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class RunLogger:
    def __init__(self, logs_dir: Path | None, *, model: str, prices: tuple[float, float] | None) -> None:
        self.path: Path | None = None
        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.path = logs_dir / f"run_{stamp}.jsonl"
            counter = 1
            while self.path.exists():
                self.path = logs_dir / f"run_{stamp}_{counter}.jsonl"
                counter += 1
        self.prices = prices
        self.totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self._write({
            "type": "run_start",
            "time": _now_iso(),
            "model": model,
            "price_per_mtok": (
                {"input": prices[0], "output": prices[1]} if prices is not None else None
            ),
        })

    def _write(self, entry: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")

    def log_attempt(
        self,
        *,
        image: str,
        attempt: int,
        request: dict[str, Any],
        response: dict[str, Any] | None,
        validation_error: str | None = None,
        transport_error: str | None = None,
    ) -> dict[str, Any]:
        """Append one attempt line; return its usage/cost for console printing."""
        usage = (response or {}).get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        cost = pricing.estimate_cost_usd(self.prices, input_tokens, output_tokens)

        if response is not None:
            self.totals["requests"] += 1
            self.totals["input_tokens"] += input_tokens
            self.totals["output_tokens"] += output_tokens
            if cost is not None:
                self.totals["cost_usd"] += cost

        if self.path is not None:
            entry: dict[str, Any] = {
                "type": "attempt",
                "time": _now_iso(),
                "image": image,
                "attempt": attempt,
                "request": {**request, "messages": sanitize_messages(request.get("messages", []))},
                "response": response,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "est_cost_usd": cost,
            }
            if validation_error is not None:
                entry["validation_error"] = validation_error
            if transport_error is not None:
                entry["transport_error"] = transport_error
            self._write(entry)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_cost_usd": cost,
        }

    def snapshot(self) -> dict[str, Any]:
        return dict(self.totals)

    def close(self) -> dict[str, Any]:
        self._write({"type": "summary", "time": _now_iso(), **self.totals})
        return self.snapshot()
