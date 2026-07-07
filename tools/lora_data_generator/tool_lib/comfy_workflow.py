"""ComfyUI UI-workflow conversion, graph audit, and API helpers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SKIP_WIDGET_INPUT_TYPES = {"IMAGEUPLOAD"}
SKIP_WIDGET_INPUT_NAMES = {"upload"}
SEED_CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}


def node_label(node: dict[str, Any]) -> str:
    return f"{node.get('id')}:{node.get('type')}"


def audit_workflow_graph(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("Workflow must contain top-level nodes and links arrays.")

    node_by_id = {}
    for node in nodes:
        node_id = node.get("id")
        if node_id in node_by_id:
            raise ValueError(f"Duplicate node id: {node_id}")
        node_by_id[node_id] = node

    link_by_id: dict[int, list[Any]] = {}
    target_edges: dict[tuple[Any, int], int] = {}
    for link in links:
        if not isinstance(link, list) or len(link) < 6:
            raise ValueError(f"Invalid ComfyUI link array: {link!r}")
        link_id, source_id, source_slot, target_id, target_slot, _link_type = link[:6]
        if link_id in link_by_id:
            raise ValueError(f"Duplicate link id: {link_id}")
        if source_id not in node_by_id:
            raise ValueError(f"Link {link_id} has missing source node {source_id}")
        if target_id not in node_by_id:
            raise ValueError(f"Link {link_id} has missing target node {target_id}")

        source_outputs = node_by_id[source_id].get("outputs") or []
        target_inputs = node_by_id[target_id].get("inputs") or []
        if not isinstance(source_slot, int) or source_slot >= len(source_outputs):
            raise ValueError(f"Link {link_id} has invalid source slot {source_slot}")
        if not isinstance(target_slot, int) or target_slot >= len(target_inputs):
            raise ValueError(f"Link {link_id} has invalid target slot {target_slot}")

        target_key = (target_id, target_slot)
        if target_key in target_edges:
            raise ValueError(
                f"Conflicting links {target_edges[target_key]} and {link_id} "
                f"both target node {target_id} input slot {target_slot}"
            )
        target_edges[target_key] = link_id

        source_type = source_outputs[source_slot].get("type")
        target_type = target_inputs[target_slot].get("type")
        if source_type != target_type and "*" not in {source_type, target_type}:
            raise ValueError(f"Link {link_id} type mismatch: {source_type!r} -> {target_type!r}")

        link_by_id[link_id] = link

    for node in nodes:
        node_id = node.get("id")
        for slot, input_info in enumerate(node.get("inputs") or []):
            link_id = input_info.get("link")
            if link_id is None:
                continue
            link = link_by_id.get(link_id)
            if link is None:
                raise ValueError(f"{node_label(node)} input {slot} references missing link {link_id}")
            if link[3] != node_id or link[4] != slot:
                raise ValueError(f"{node_label(node)} input {slot} conflicts with top-level link {link_id}")

        for slot, output_info in enumerate(node.get("outputs") or []):
            for link_id in output_info.get("links") or []:
                link = link_by_id.get(link_id)
                if link is None:
                    raise ValueError(f"{node_label(node)} output {slot} references missing link {link_id}")
                if link[1] != node_id or link[2] != slot:
                    raise ValueError(f"{node_label(node)} output {slot} conflicts with top-level link {link_id}")


def widget_value_stream(node: dict[str, Any], widget_input_count: int) -> list[Any]:
    values = list(node.get("widgets_values") or [])
    if len(values) == widget_input_count + 1 and len(values) > 1 and values[1] in SEED_CONTROL_VALUES:
        values.pop(1)
    return values


def should_skip_widget_input(input_info: dict[str, Any]) -> bool:
    return (
        input_info.get("type") in SKIP_WIDGET_INPUT_TYPES
        or input_info.get("name") in SKIP_WIDGET_INPUT_NAMES
    )


def convert_ui_workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    links = {
        link[0]: [str(link[1]), link[2]]
        for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 6
    }

    prompt: dict[str, Any] = {}
    for ui_node in workflow.get("nodes", []):
        node_id = str(ui_node["id"])
        inputs: dict[str, Any] = {}
        ui_inputs = ui_node.get("inputs") or []
        widget_inputs = [inp for inp in ui_inputs if "widget" in inp]
        widget_values = widget_value_stream(ui_node, len(widget_inputs))
        widget_index = 0

        for input_info in ui_inputs:
            name = input_info.get("name")
            if not name:
                continue

            widget_value: Any | None = None
            has_widget = "widget" in input_info
            if has_widget:
                if widget_index >= len(widget_values):
                    raise ValueError(
                        f"Node {node_id} ({ui_node.get('type')}) has fewer widget values than widget inputs."
                    )
                widget_value = widget_values[widget_index]
                widget_index += 1

            link_id = input_info.get("link")
            if link_id is not None:
                if link_id not in links:
                    raise ValueError(f"Node {node_id} ({ui_node.get('type')}) references missing link {link_id}.")
                inputs[name] = links[link_id]
            elif has_widget and not should_skip_widget_input(input_info):
                inputs[name] = widget_value

        prompt[node_id] = {
            "class_type": ui_node["type"],
            "inputs": inputs,
            "_meta": {"title": ui_node.get("title") or ui_node["type"]},
        }

    return prompt


def post_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {exc.code}: {body}") from exc


def get_json(base_url: str, endpoint: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + endpoint, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(base_url: str, endpoint: str, query: dict[str, str]) -> bytes:
    url = base_url.rstrip("/") + endpoint + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def wait_for_history(base_url: str, prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = get_json(base_url, f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def history_output_images(history: dict[str, Any]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for output in (history.get("outputs") or {}).values():
        for image in output.get("images") or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(
                    {
                        "filename": str(image.get("filename", "")),
                        "subfolder": str(image.get("subfolder", "")),
                        "type": str(image.get("type", "output")),
                    }
                )
    return images


def read_history_image(base_url: str, image: dict[str, str], repo_output_dir: Path) -> bytes:
    if image.get("type") == "output":
        path = repo_output_dir / image.get("subfolder", "") / image["filename"]
        if path.is_file():
            return path.read_bytes()
    return get_bytes(
        base_url,
        "/view",
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        },
    )
