"""Input job JSON loading, normalization, and field access shared by both pipelines.

An input JSON file may be a single job object, a list of job objects, or an
object containing an "items", "jobs", or "prompts" list. An input directory is
expanded as sorted direct child .json files. Prompt fields may also be nested
under a per-job "chunks" object. See the tool README for the full format.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class JobItem:
    item: dict[str, Any]
    source_path: Path
    base_dir: Path
    input_stem: str


def input_json_files(path: Path | Sequence[Path], source_kind: str) -> list[Path]:
    if source_kind == "file":
        if not isinstance(path, Path):
            raise ValueError("--input-json expects exactly one JSON file path.")
        if path.is_file():
            return [path]
        if path.is_dir():
            raise IsADirectoryError(f"--input-json expects a JSON file, got a directory. Use --input-dir instead: {path}")
        raise FileNotFoundError(f"Input JSON file not found: {path}")

    if source_kind == "dir":
        if not isinstance(path, Path):
            raise ValueError("--input-dir expects exactly one directory path.")
        if path.is_file():
            raise NotADirectoryError(f"--input-dir expects a directory, got a file. Use --input-json instead: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Input JSON directory not found: {path}")
        files = sorted(path.glob("*.json"), key=lambda item: item.name.lower())
        if not files:
            raise ValueError(f"Input directory contains no .json files: {path}")
        return files

    if source_kind == "files":
        if isinstance(path, Path):
            paths = [path]
        else:
            paths = list(path)
        if not paths:
            raise ValueError("--input-files requires at least one JSON file.")
        for item in paths:
            if item.is_dir():
                raise IsADirectoryError(f"--input-files expects JSON files, got a directory: {item}")
            if not item.is_file():
                raise FileNotFoundError(f"Input JSON file not found: {item}")
        return paths

    raise ValueError(f"Unknown input source kind: {source_kind}")


def load_job_items(path: Path | Sequence[Path], limit: int | None = None, *, source_kind: str = "file") -> list[JobItem]:
    jobs: list[JobItem] = []
    for source_path in input_json_files(path, source_kind):
        data = json.loads(source_path.read_text(encoding="utf-8"))
        for item in normalize_items(data):
            jobs.append(
                JobItem(
                    item=item,
                    source_path=source_path,
                    base_dir=source_path.parent,
                    input_stem=source_path.stem,
                )
            )
            if limit is not None and len(jobs) >= limit:
                return jobs
    return jobs


def input_collection_stem(path: Path | Sequence[Path]) -> str:
    if isinstance(path, Path):
        return clean_stem(path.stem)
    paths = list(path)
    if not paths:
        return "input"
    if len(paths) == 1:
        return clean_stem(paths[0].stem)
    return clean_stem(f"selected_{len(paths)}_files_{paths[0].stem}")


def normalize_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("items", "jobs", "prompts"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            items = [data]
    else:
        raise ValueError("Input JSON must be an object, list, or object containing items/jobs/prompts.")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be a JSON object.")
        normalized.append(item)
    return normalized


def get_field(item: dict[str, Any], *names: str, default: Any = "") -> Any:
    chunks = item.get("chunks")
    for source in (item, chunks if isinstance(chunks, dict) else {}):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def list_field(item: dict[str, Any], *names: str) -> list[Any]:
    value = get_field(item, *names, default=[])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Expected one of {names} to be a list, got {type(value).__name__}")
    return value


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def clean_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "image"


def item_stem(item: dict[str, Any], index: int) -> str:
    value = get_field(item, "output_stem", "name", "id", default=f"item_{index:05d}")
    return clean_stem(str(value))


def output_prefix(base_prefix: str, stem: str, timestamp_s: int, repeat_index: int, repeat_count: int) -> str:
    repeat_suffix = f"_r{repeat_index:02d}" if repeat_count > 1 else ""
    return f"{base_prefix.strip('/')}/{stem}{repeat_suffix}_{timestamp_s}"


def image_output_name(input_stem: str, backend: str, timestamp_s: int) -> str:
    """Final image filename pattern shared by both pipelines:
    {raw_json_filename}_{flux2|gpt|klein}_{timestamp in seconds}."""
    return f"{clean_stem(input_stem)}_{backend}_{timestamp_s}"


SUFFIX_BY_FORMAT = {"png": ".png", "jpeg": ".jpeg", "webp": ".webp"}
