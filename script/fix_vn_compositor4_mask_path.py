import json
from pathlib import Path


WORKFLOW = Path("user/default/workflows/vn_foundation_environment_person_flux2_layout.json")


def node_map(data):
    return {int(n["id"]): n for n in data.get("nodes", [])}


def by_type(data, node_type):
    return [n for n in data.get("nodes", []) if n.get("type") == node_type]


def socket_index(sockets, name, fallback):
    for index, socket in enumerate(sockets or []):
        if socket.get("name") == name:
            return index
    return fallback


def out(name, typ, slot_index, links=None):
    return {
        "localized_name": name,
        "name": name,
        "type": typ,
        "slot_index": slot_index,
        "links": links[:] if links else None,
    }


def inp(name, typ, link=None):
    return {
        "localized_name": name,
        "name": name,
        "type": typ,
        "link": link,
    }


def clean_link_refs(data, link_ids):
    link_ids = {str(x) for x in link_ids}
    for node in data.get("nodes", []):
        for socket in node.get("inputs") or []:
            if str(socket.get("link")) in link_ids:
                socket["link"] = None
        for socket in node.get("outputs") or []:
            links = socket.get("links") or []
            socket["links"] = [x for x in links if str(x) not in link_ids] or None


def remove_links_to_inputs(data, targets):
    targets = {(int(node_id), int(slot)) for node_id, slot in targets}
    removed = []
    kept = []
    for row in data.get("links", []):
        if len(row) >= 5 and (int(row[3]), int(row[4])) in targets:
            removed.append(row[0])
        else:
            kept.append(row)
    data["links"] = kept
    clean_link_refs(data, removed)


def remove_links_from_outputs(data, sources):
    sources = {(int(node_id), int(slot)) for node_id, slot in sources}
    removed = []
    kept = []
    for row in data.get("links", []):
        if len(row) >= 3 and (int(row[1]), int(row[2])) in sources:
            removed.append(row[0])
        else:
            kept.append(row)
    data["links"] = kept
    clean_link_refs(data, removed)


def next_node_id(data):
    used = {int(n["id"]) for n in data.get("nodes", [])}
    value = max(used, default=0) + 1
    while value in used:
        value += 1
    return value


def next_link_factory(data):
    used = {int(row[0]) for row in data.get("links", [])}
    value = max(used, default=0) + 1

    def next_link(preferred=None):
        nonlocal value
        if preferred is not None and preferred not in used:
            used.add(preferred)
            return preferred
        while value in used:
            value += 1
        result = value
        used.add(result)
        value += 1
        return result

    return next_link


def ensure_masks_node(data, comp):
    existing = by_type(data, "Compositor4MasksOutput")
    if existing:
        node = existing[0]
    else:
        node = {
            "id": next_node_id(data),
            "type": "Compositor4MasksOutput",
            "pos": [1760, -125],
            "size": [320, 390],
            "flags": {},
            "order": 25,
            "mode": 0,
            "inputs": [inp("layer_outputs", "COMPOSITOR_OUTPUT_MASKS")],
            "outputs": [],
            "properties": {"Node name for S&R": "Compositor4MasksOutput"},
            "widgets_values": [],
        }
        data["nodes"].append(node)

    node["inputs"] = [inp("layer_outputs", "COMPOSITOR_OUTPUT_MASKS", node.get("inputs", [{}])[0].get("link"))]
    image_outputs = [out(f"image_{i}", "IMAGE", i - 1, None) for i in range(1, 9)]
    mask_outputs = [out(f"mask_{i}", "MASK", 7 + i, None) for i in range(1, 9)]
    node["outputs"] = image_outputs + mask_outputs
    node["properties"] = {"Node name for S&R": "Compositor4MasksOutput"}

    while len(comp.get("outputs") or []) < 4:
        comp.setdefault("outputs", []).append(out("layer_outputs", "COMPOSITOR_OUTPUT_MASKS", len(comp["outputs"])))
    comp["outputs"][3] = out("layer_outputs", "COMPOSITOR_OUTPUT_MASKS", 3, comp["outputs"][3].get("links"))
    return node


def ensure_final_invert(data, switch, switch_mask_slot):
    switch_link = switch["inputs"][switch_mask_slot].get("link")
    nodes = node_map(data)
    if switch_link is not None:
        for row in data.get("links", []):
            if row[0] == switch_link:
                candidate = nodes.get(int(row[1]))
                if candidate and candidate.get("type") == "Mask Invert":
                    return candidate

    node = {
        "id": next_node_id(data),
        "type": "Mask Invert",
        "pos": [1500, 95],
        "size": [210, 58],
        "flags": {},
        "order": 24,
        "mode": 0,
        "inputs": [inp("mask", "MASK")],
        "outputs": [out("MASK", "MASK", 0, None)],
        "properties": {"Node name for S&R": "Mask Invert"},
        "widgets_values": [],
    }
    data["nodes"].append(node)
    return node


def ensure_config_mask_invert(data, config, config_mask_slot):
    config_link = config["inputs"][config_mask_slot].get("link")
    nodes = node_map(data)
    if config_link is not None:
        for row in data.get("links", []):
            if row[0] == config_link:
                candidate = nodes.get(int(row[1]))
                if candidate and candidate.get("type") == "Mask Invert":
                    return candidate

    node = {
        "id": next_node_id(data),
        "type": "Mask Invert",
        "pos": [640, -35],
        "size": [210, 58],
        "flags": {},
        "order": 12,
        "mode": 0,
        "inputs": [inp("masks", "MASK")],
        "outputs": [out("MASKS", "MASK", 0, None)],
        "properties": {"Node name for S&R": "Mask Invert"},
        "widgets_values": [],
    }
    data["nodes"].append(node)
    return node


def audit(data):
    nodes = {str(n["id"]): n for n in data.get("nodes", [])}
    errors = []
    seen = set()
    targets = set()
    for row in data.get("links", []):
        if not isinstance(row, list) or len(row) < 6:
            errors.append(f"malformed link row: {row!r}")
            continue
        link_id, src_id, src_slot, dst_id, dst_slot, typ = row[:6]
        if link_id in seen:
            errors.append(f"duplicate link id {link_id}")
        seen.add(link_id)
        src = nodes.get(str(src_id))
        dst = nodes.get(str(dst_id))
        if src is None or dst is None:
            errors.append(f"dangling node ref link {link_id}: {src_id}->{dst_id}")
            continue
        outs = src.get("outputs") or []
        ins = dst.get("inputs") or []
        if int(src_slot) >= len(outs):
            errors.append(f"missing source socket link {link_id}: {src_id}:{src_slot}")
            continue
        if int(dst_slot) >= len(ins):
            errors.append(f"missing target socket link {link_id}: {dst_id}:{dst_slot}")
            continue
        target = (str(dst_id), int(dst_slot))
        if target in targets:
            errors.append(f"conflicting target input {target}")
        targets.add(target)
        if str(link_id) not in {str(x) for x in (outs[int(src_slot)].get("links") or [])}:
            errors.append(f"source output missing link {link_id}")
        if str(ins[int(dst_slot)].get("link")) != str(link_id):
            errors.append(f"target input missing link {link_id}")
        out_type = str(outs[int(src_slot)].get("type", "")).upper()
        in_type = str(ins[int(dst_slot)].get("type", "")).upper()
        row_type = str(typ).upper()
        if out_type and in_type and out_type != in_type:
            errors.append(f"type mismatch link {link_id}: {out_type}->{in_type}")
        if row_type and out_type and row_type != out_type:
            errors.append(f"row/source type mismatch link {link_id}: {row_type}!={out_type}")
    return errors


def main():
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    comps = by_type(data, "Compositor4")
    configs = by_type(data, "CompositorConfig4")
    switches = by_type(data, "ImageMaskSwitch")
    load_images = by_type(data, "LoadImage")
    if not comps:
        raise SystemExit("No Compositor4 node found")
    if not configs:
        raise SystemExit("No CompositorConfig4 node found")
    if not switches:
        raise SystemExit("No ImageMaskSwitch node found")

    comp = comps[0]
    config = configs[0]
    switch = switches[0]
    person_loader = next((n for n in load_images if int(n["id"]) == 5), None)
    if person_loader is None:
        raise SystemExit("No LoadImage node 5 found for person image")

    config_mask2_slot = socket_index(config.get("inputs"), "mask2", 3)
    switch_mask_slot = socket_index(switch.get("inputs"), "mask1", 1)
    masks_node = ensure_masks_node(data, comp)
    config_mask_invert = ensure_config_mask_invert(data, config, config_mask2_slot)
    final_invert = ensure_final_invert(data, switch, switch_mask_slot)
    config_invert_input_slot = socket_index(config_mask_invert.get("inputs"), "masks", 0)
    final_input_slot = socket_index(final_invert.get("inputs"), "mask", 0)

    mask2_output_slot = socket_index(masks_node.get("outputs"), "mask_2", 9)
    next_link = next_link_factory(data)
    link_loader_mask_to_config_invert = next_link(116)
    link_config_invert_to_config = next_link(117)
    link_comp_to_masks = next_link(111)
    link_mask_to_invert = next_link(112)
    link_invert_to_switch = next_link(113)

    remove_links_to_inputs(
        data,
        [
            (config_mask_invert["id"], config_invert_input_slot),
            (config["id"], config_mask2_slot),
            (masks_node["id"], 0),
            (final_invert["id"], final_input_slot),
            (switch["id"], switch_mask_slot),
        ],
    )
    remove_links_from_outputs(
        data,
        [
            (person_loader["id"], 1),
            (config_mask_invert["id"], 0),
            (comp["id"], 3),
            (masks_node["id"], mask2_output_slot),
            (final_invert["id"], 0),
        ],
    )

    person_loader["outputs"][1]["links"] = [link_loader_mask_to_config_invert]
    config_mask_invert["inputs"][config_invert_input_slot]["link"] = link_loader_mask_to_config_invert
    config_mask_invert["outputs"][0]["links"] = [link_config_invert_to_config]
    config["inputs"][config_mask2_slot]["link"] = link_config_invert_to_config
    comp["outputs"][3]["links"] = [link_comp_to_masks]
    masks_node["inputs"][0]["link"] = link_comp_to_masks
    masks_node["outputs"][mask2_output_slot]["links"] = [link_mask_to_invert]
    final_invert["inputs"][final_input_slot]["link"] = link_mask_to_invert
    final_invert["outputs"][0]["links"] = [link_invert_to_switch]
    switch["inputs"][switch_mask_slot]["link"] = link_invert_to_switch

    data["links"].extend(
        [
            [link_loader_mask_to_config_invert, person_loader["id"], 1, config_mask_invert["id"], config_invert_input_slot, "MASK"],
            [link_config_invert_to_config, config_mask_invert["id"], 0, config["id"], config_mask2_slot, "MASK"],
            [link_comp_to_masks, comp["id"], 3, masks_node["id"], 0, "COMPOSITOR_OUTPUT_MASKS"],
            [link_mask_to_invert, masks_node["id"], mask2_output_slot, final_invert["id"], final_input_slot, "MASK"],
            [link_invert_to_switch, final_invert["id"], 0, switch["id"], switch_mask_slot, "MASK"],
        ]
    )

    data["last_node_id"] = max(int(data.get("last_node_id", 0)), *(int(n["id"]) for n in data["nodes"]))
    data["last_link_id"] = max(int(data.get("last_link_id", 0)), *(int(row[0]) for row in data["links"]))

    errors = audit(data)
    if errors:
        print("Graph audit failed; not saving")
        for error in errors:
            print(error)
        raise SystemExit(1)

    WORKFLOW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Graph audit ok")
    print(
        "Mask path: "
        f"LoadImage({person_loader['id']}) MASK -> "
        f"Mask Invert({config_mask_invert['id']}) -> "
        f"CompositorConfig4({config['id']}) mask2; "
        f"Compositor4({comp['id']}) layer_outputs -> "
        f"Compositor4MasksOutput({masks_node['id']}) mask_2 -> "
        f"Mask Invert({final_invert['id']}) -> "
        f"ImageMaskSwitch({switch['id']}) mask1"
    )


if __name__ == "__main__":
    main()
