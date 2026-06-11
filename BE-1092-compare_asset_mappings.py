#!/usr/bin/env python3
"""Compare FE MODEL_NODE_MAPPINGS against Core /object_info and /api/assets."""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_FRONTEND = Path(
    "/home/simon/comfy/ComfyUI_frontend/.wt/dante01yoon/"
    "pr-12411-integration-do-not-merge-m1-fe-asset-sta"
)
DEFAULT_MAPPING = DEFAULT_FRONTEND / "src/platform/assets/mappings/modelNodeMappings.ts"


def fetch_json(base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    url = base_url.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def fetch_object_info(base_url: str) -> dict[str, Any]:
    try:
        return fetch_json(base_url, "/object_info")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return fetch_json(base_url, "/api/object_info")


def fetch_all_model_assets(base_url: str) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    offset = 0
    limit = 500
    while True:
        page = fetch_json(
            base_url,
            "/api/assets",
            {"asset_type": "model", "limit": limit, "offset": offset},
        )
        batch = page.get("assets", [])
        assets.extend(batch)
        if not page.get("has_more") or not batch:
            return assets
        offset += len(batch)


def parse_model_node_mappings(path: Path) -> list[tuple[str, str, str]]:
    text = path.read_text()
    pattern = re.compile(
        r"\[\s*(['\"])(.*?)\1\s*,\s*(['\"])(.*?)\3\s*,\s*(['\"])(.*?)\5\s*\]",
        re.DOTALL,
    )
    return [(m.group(2), m.group(4), m.group(6)) for m in pattern.finditer(text)]


def get_combo_options(node_def: dict[str, Any], input_key: str) -> list[str] | None:
    inputs = node_def.get("input", {})
    for section in ("required", "optional"):
        spec = inputs.get(section, {}).get(input_key)
        if spec is None:
            continue
        if not isinstance(spec, list) or not spec:
            return None
        input_type = spec[0]
        options = (
            spec[1].get("options")
            if len(spec) > 1 and isinstance(spec[1], dict)
            else None
        )
        if isinstance(input_type, list):
            return [str(item) for item in input_type]
        if input_type == "COMBO" and isinstance(options, list):
            return [str(item) for item in options]
        return None
    return None


def asset_values_by_folder(assets: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for asset in assets:
        display_name = asset.get("display_name") or asset.get("name")
        if not display_name:
            continue
        folders = asset.get("model_folders") or []
        if not folders and asset.get("model_folder"):
            folders = [asset["model_folder"]]
        for folder in folders:
            values[str(folder)].add(str(display_name))
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:6410")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args()

    mappings = parse_model_node_mappings(args.mapping)
    try:
        object_info = fetch_object_info(args.base_url)
        assets = fetch_all_model_assets(args.base_url)
    except urllib.error.URLError as error:
        print(f"Failed to reach {args.base_url}: {error}", file=sys.stderr)
        return 2

    assets_by_folder = asset_values_by_folder(assets)
    rows: list[dict[str, Any]] = []

    for model_folder, node_class, input_key in mappings:
        node_def = object_info.get(node_class)
        if not node_def:
            rows.append(
                {
                    "status": "missing_node",
                    "model_folder": model_folder,
                    "node_class": node_class,
                    "input_key": input_key,
                }
            )
            continue
        if not input_key:
            rows.append(
                {
                    "status": "no_input_key",
                    "model_folder": model_folder,
                    "node_class": node_class,
                    "input_key": input_key,
                }
            )
            continue
        options = get_combo_options(node_def, input_key)
        if options is None:
            rows.append(
                {
                    "status": "missing_or_non_combo_input",
                    "model_folder": model_folder,
                    "node_class": node_class,
                    "input_key": input_key,
                }
            )
            continue

        object_values = set(options)
        asset_values = assets_by_folder.get(model_folder, set())
        rows.append(
            {
                "status": "match" if object_values == asset_values else "diff",
                "model_folder": model_folder,
                "node_class": node_class,
                "input_key": input_key,
                "object_info_count": len(object_values),
                "asset_count": len(asset_values),
                "missing_from_assets": sorted(object_values - asset_values),
                "extra_in_assets": sorted(asset_values - object_values),
            }
        )

    statuses = dict(
        sorted(
            (status, sum(1 for r in rows if r["status"] == status))
            for status in {r["status"] for r in rows}
        )
    )
    summary = {
        "mapping_file": str(args.mapping),
        "base_url": args.base_url,
        "mapping_rows": len(mappings),
        "model_assets": len(assets),
        "asset_folders": sorted(assets_by_folder),
        "statuses": statuses,
        "rows": rows,
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Mapping file: {summary['mapping_file']}")
    print(f"Base URL: {args.base_url}")
    print(f"Mappings: {len(mappings)}  model assets: {len(assets)}")
    print(f"Asset folders: {', '.join(summary['asset_folders']) or '(none)'}")
    print("Statuses: " + ", ".join(f"{k}={v}" for k, v in statuses.items()))
    print()

    for row in rows:
        if row["status"] == "match":
            print(
                f"MATCH {row['model_folder']} -> {row['node_class']}.{row['input_key']} ({row['asset_count']})"
            )
        elif row["status"] == "diff":
            print(
                f"DIFF  {row['model_folder']} -> {row['node_class']}.{row['input_key']} "
                f"object_info={row['object_info_count']} assets={row['asset_count']}"
            )
            if row["missing_from_assets"]:
                print(
                    "  missing_from_assets: "
                    + ", ".join(row["missing_from_assets"][:10])
                )
            if row["extra_in_assets"]:
                print("  extra_in_assets: " + ", ".join(row["extra_in_assets"][:10]))
        else:
            print(
                f"{row['status'].upper()} {row['model_folder']} -> {row['node_class']}.{row['input_key']}"
            )

    return 1 if any(row["status"] == "diff" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
