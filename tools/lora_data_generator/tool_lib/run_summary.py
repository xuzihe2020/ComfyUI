"""Run-level summary logging shared by generation backends."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now_s() -> int:
    return int(time.time())


def elapsed_s(started_perf: float) -> float:
    return round(time.perf_counter() - started_perf, 3)


def average_s(total_s: float, count: int) -> float | None:
    if count < 1:
        return None
    return round(total_s / count, 3)


def write_run_summary(log_dir: Path, mode: str, run_timestamp_s: int, payload: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"run_summary_{mode}_{run_timestamp_s}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
