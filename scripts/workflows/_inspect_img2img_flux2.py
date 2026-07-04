#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


WORKFLOW = Path("user/default/workflows/example/img2img_flux_2.json")


def main() -> None:
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    for node in workflow.get("nodes", []):
        title = node.get("title")
        node_type = node.get("type")
        properties = node.get("properties") or {}
        sr_name = properties.get("Node name for S&R")
        if node_type in {
            "LoadImage",
            "SaveImage",
            "CLIPTextEncode",
            "PrimitiveNode",
            "PrimitiveInt",
            "RandomNoise",
        } or title in {"width", "height", "denoise"}:
            print(f"id={node.get('id')} type={node_type!r} title={title!r} sr={sr_name!r}")
            print(f"  widgets={node.get('widgets_values')!r}")
            print("  inputs:")
            for input_info in node.get("inputs") or []:
                print(
                    "   - "
                    f"name={input_info.get('name')!r} type={input_info.get('type')!r} "
                    f"widget={(input_info.get('widget') or {}).get('name')!r} "
                    f"link={input_info.get('link')!r}"
                )
            print("  outputs:")
            for output_info in node.get("outputs") or []:
                print(
                    "   - "
                    f"name={output_info.get('name')!r} type={output_info.get('type')!r} "
                    f"links={output_info.get('links')!r}"
                )


if __name__ == "__main__":
    main()
