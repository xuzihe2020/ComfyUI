from pathlib import Path
import json

from build_seven_stage_flux2_workflow import (
    Workflow,
    add_da3_depth_fill,
    add_image_scale,
    add_load_image,
    add_mask_preview,
    add_person_mask,
    add_preview,
    add_save,
    audit,
)


OUTS = [
    Path("user/default/workflows/dev/vn_step1_preprocess_a_mask_depth_fill.json"),
    Path("user/default/workflows/dev/vn_step1_preprocess_a_masks_depth_canny.json"),
]


def build():
    wf = Workflow()
    wf.group("1. Input A and fixed final canvas", -900, -520, 900, 760, "#343")
    wf.group("2. Auto person mask with YOLO + SAM", 120, -520, 2000, 760, "#334")
    wf.group("3. Depth Anything 3 raw depth + background-depth fill", 120, 420, 2460, 820, "#443")

    a = add_load_image(wf, "A original composition image", -860, -400, "example.png")
    scaled_a = add_image_scale(wf, a, -500, -360)
    add_preview(wf, scaled_a, 0, -860, -20, "A fixed canvas preview")
    add_save(wf, scaled_a, 0, -500, -20, "vn_step1_preprocess/a_fixed_canvas", "save fixed A")

    auto_person, person_expanded, person_feather = add_person_mask(wf, scaled_a, 160, -360)
    add_mask_preview(wf, auto_person, 0, 160, 500, "auto person mask raw")
    add_mask_preview(wf, person_expanded, 0, 160, 880, "person mask expanded")
    add_mask_preview(wf, person_feather, 0, 780, 500, "person mask feathered")
    raw_depth, bg_depth_fill = add_da3_depth_fill(wf, scaled_a, person_feather, 120, 500, "A")
    add_preview(wf, raw_depth, 0, 1660, 470, "raw depth preview")
    add_preview(wf, bg_depth_fill, 0, 1660, 820, "BACKGROUND-DEPTH FILL preview")
    add_save(wf, bg_depth_fill, 0, 2080, 820, "vn_step1_preprocess/background_depth_fill", "save background-depth fill")

    data = {
        "id": "vn_step1_preprocess_a_mask_depth_fill",
        "revision": 0,
        "last_node_id": wf.next_node_id - 1,
        "last_link_id": wf.next_link_id - 1,
        "nodes": wf.nodes,
        "links": wf.links,
        "groups": wf.groups,
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [760, 410]}},
        "version": 0.4,
    }
    return data


def main():
    data = build()
    audit(data)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    for out in OUTS:
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out}")
    print(f"nodes={len(data['nodes'])} links={len(data['links'])} groups={len(data['groups'])}")


if __name__ == "__main__":
    main()
