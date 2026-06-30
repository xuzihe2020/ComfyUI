import json
from pathlib import Path


WORKFLOW = Path("user/default/workflows/vn_foundation_environment_person_flux2_layout.json")


def node_map(data):
    return {int(n["id"]): n for n in data.get("nodes", [])}


def out(name, typ, slot_index, links=None):
    return {
        "localized_name": name,
        "name": name,
        "type": typ,
        "slot_index": slot_index,
        "links": links,
    }


def set_output_links(node, slot, links):
    node["outputs"][slot]["links"] = links[:] if links else None


def remove_link_from_output(node, slot, link_id):
    links = node["outputs"][slot].get("links") or []
    links = [x for x in links if x != link_id]
    node["outputs"][slot]["links"] = links or None


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
        if src_slot >= len(outs):
            errors.append(f"missing source socket link {link_id}: {src_id}:{src_slot}")
            continue
        if dst_slot >= len(ins):
            errors.append(f"missing target socket link {link_id}: {dst_id}:{dst_slot}")
            continue
        target = (str(dst_id), int(dst_slot))
        if target in targets:
            errors.append(f"conflicting target input {target}")
        targets.add(target)
        if str(link_id) not in {str(x) for x in (outs[src_slot].get("links") or [])}:
            errors.append(f"source output missing link {link_id}")
        if str(ins[dst_slot].get("link")) != str(link_id):
            errors.append(f"target input missing link {link_id}")
        out_type = str(outs[src_slot].get("type", "")).upper()
        in_type = str(ins[dst_slot].get("type", "")).upper()
        row_type = str(typ).upper()
        if out_type and in_type and out_type != in_type:
            errors.append(f"type mismatch link {link_id}: {out_type}->{in_type}")
        if row_type and out_type and row_type != out_type:
            errors.append(f"row/source type mismatch link {link_id}: {row_type}!={out_type}")
    return errors


def main():
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = node_map(data)
    links = data["links"]

    comp = nodes[11]
    switch = nodes[15]
    raw_invert = nodes[52]
    final_invert = nodes[62]

    next_node = max(nodes) + 1
    masks_node_id = 63 if 63 not in nodes else next_node
    existing_link_ids = {int(row[0]) for row in links}
    next_link = max(existing_link_ids) + 1

    def new_link_id(preferred=None):
        nonlocal next_link
        if preferred is not None and preferred not in existing_link_ids:
            existing_link_ids.add(preferred)
            return preferred
        while next_link in existing_link_ids:
            next_link += 1
        value = next_link
        existing_link_ids.add(value)
        next_link += 1
        return value

    link_comp_to_masks = new_link_id(111)
    link_mask_to_invert = new_link_id(112)
    link_invert_to_switch = 105 if 105 in existing_link_ids else new_link_id(105)

    # Remove stale raw-image2-mask -> final invert link. That was the portrait-sized path.
    links[:] = [row for row in links if row[0] not in {21, link_mask_to_invert, link_comp_to_masks}]
    remove_link_from_output(raw_invert, 0, 21)
    final_invert["inputs"][0]["link"] = link_mask_to_invert
    final_invert["outputs"][0]["links"] = [link_invert_to_switch]
    switch["inputs"][1]["link"] = link_invert_to_switch

    # Ensure Compositor4 exposes and links its V4 layer output.
    while len(comp["outputs"]) < 4:
        comp["outputs"].append(out("layer_outputs", "COMPOSITOR_OUTPUT_MASKS", len(comp["outputs"]), None))
    comp["outputs"][3] = out("layer_outputs", "COMPOSITOR_OUTPUT_MASKS", 3, [link_comp_to_masks])

    # Add/replace the V4 masks unpacker.
    data["nodes"] = [n for n in data["nodes"] if int(n["id"]) != masks_node_id]
    image_outputs = [out(f"image_{i}", "IMAGE", i - 1, None) for i in range(1, 9)]
    mask_outputs = [out(f"mask_{i}", "MASK", 7 + i, [link_mask_to_invert] if i == 2 else None) for i in range(1, 9)]
    data["nodes"].append(
        {
            "id": masks_node_id,
            "type": "Compositor4MasksOutput",
            "pos": [1760, -125],
            "size": [320, 390],
            "flags": {},
            "order": 25,
            "mode": 0,
            "inputs": [
                {
                    "localized_name": "layer_outputs",
                    "name": "layer_outputs",
                    "type": "COMPOSITOR_OUTPUT_MASKS",
                    "link": link_comp_to_masks,
                }
            ],
            "outputs": image_outputs + mask_outputs,
            "properties": {"Node name for S&R": "Compositor4MasksOutput"},
            "widgets_values": [],
        }
    )

    links[:] = [row for row in links if row[0] not in {link_invert_to_switch, link_comp_to_masks, link_mask_to_invert}]
    links.extend(
        [
            [link_invert_to_switch, 62, 0, 15, 1, "MASK"],
            [link_comp_to_masks, 11, 3, masks_node_id, 0, "COMPOSITOR_OUTPUT_MASKS"],
            [link_mask_to_invert, masks_node_id, 9, 62, 0, "MASK"],
        ]
    )

    data["last_node_id"] = max(int(data.get("last_node_id", 0)), masks_node_id)
    data["last_link_id"] = max(int(data.get("last_link_id", 0)), max(int(row[0]) for row in links))

    errors = audit(data)
    if errors:
        print("Graph audit failed; not saving")
        for error in errors:
            print(error)
        raise SystemExit(1)

    WORKFLOW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Graph audit ok")
    print(f"Mask path: Compositor4 layer_outputs -> Compositor4MasksOutput mask_2 -> node 62 invert -> ImageMaskSwitch -> preview/inpaint")


if __name__ == "__main__":
    main()
