from pathlib import Path
import json

from build_seven_stage_flux2_workflow import (
    Workflow,
    add_da3_depth_fill,
    add_flux_inpaint_stage,
    add_image_scale,
    add_load_image,
    add_mask_preview,
    add_person_mask,
    add_preview,
    add_save,
    audit,
)


OUT = Path("user/default/workflows/dev/vn_step1_step2_background_plate_flux2.json")


def add_flux2_loaders(wf):
    wf.group("0. FLUX.2 model loaders", -900, -980, 900, 380, "#433")
    unet = wf.node(
        "UNETLoader",
        -860,
        -920,
        [wf.inp("unet_name", "COMBO", True), wf.inp("weight_dtype", "COMBO", True)],
        [wf.out("MODEL", "MODEL")],
        ["flux2\\flux_2_klein_9b.safetensors", "default"],
        title="FLUX.2 Klein/Dev model",
        size=[340, 82],
        color="#322",
        bgcolor="#533",
    )
    clip = wf.node(
        "CLIPLoader",
        -500,
        -920,
        [
            wf.inp("clip_name", "COMBO", True),
            wf.inp("type", "COMBO", True),
            wf.inp("device", "COMBO", True),
        ],
        [wf.out("CLIP", "CLIP")],
        ["flux2\\qwen_3_8b_fp8mixed.safetensors", "flux2", "default"],
        title="FLUX.2 text encoder",
        size=[340, 106],
        color="#322",
        bgcolor="#533",
    )
    vae = wf.node(
        "VAELoader",
        -140,
        -920,
        [wf.inp("vae_name", "COMBO", True)],
        [wf.out("VAE", "VAE")],
        ["flux2\\full_encoder_small_decoder.safetensors"],
        title="FLUX.2 VAE",
        size=[340, 58],
        color="#322",
        bgcolor="#533",
    )
    return unet, clip, vae


def build():
    wf = Workflow()
    unet, clip, vae = add_flux2_loaders(wf)

    wf.group("1. Input A and fixed final canvas", -900, -520, 900, 760, "#343")
    wf.group("2. Auto person mask with YOLO + SAM", 120, -520, 2000, 760, "#334")
    wf.group("3. Step 1 background-depth fill from A", 120, 420, 2460, 820, "#443")
    wf.group("4. Environment references for Step 2", -900, 420, 900, 980, "#343")
    wf.group("5. Step 2 FLUX.2 inpaint: empty background plate", 2820, -520, 3740, 1180, "#443")

    a = add_load_image(wf, "A original composition image", -860, -400, "example.png")
    scaled_a = add_image_scale(wf, a, -500, -360)
    add_preview(wf, scaled_a, 0, -860, -20, "A fixed canvas preview")
    add_save(wf, scaled_a, 0, -500, -20, "vn_step1_step2/a_fixed_canvas", "save fixed A")

    auto_person, person_expanded, person_feather = add_person_mask(wf, scaled_a, 160, -360)

    add_mask_preview(wf, auto_person, 0, 160, 500, "auto person mask raw")
    add_mask_preview(wf, person_expanded, 0, 160, 880, "person mask expanded")
    add_mask_preview(wf, person_feather, 0, 780, 500, "STEP 2 INPAINT MASK - feathered person")

    raw_depth, bg_depth_fill = add_da3_depth_fill(wf, scaled_a, person_feather, 120, 500, "A")
    add_preview(wf, raw_depth, 0, 1660, 470, "raw depth debug only")
    add_preview(wf, bg_depth_fill, 0, 1660, 820, "STEP 2 STRUCTURE REF - background-depth fill")
    add_save(wf, bg_depth_fill, 0, 2080, 820, "vn_step1_step2/background_depth_fill", "save background-depth fill")

    env1 = add_load_image(wf, "Environment reference #1 - replace before queue", -860, 520, "example.png")
    env2 = add_load_image(wf, "Environment reference #2 - replace before queue", -520, 520, "example.png")
    env3 = add_load_image(wf, "Environment reference #3 - replace before queue", -860, 900, "example.png")

    prompt = (
        "Generate an empty background plate with no people. Use image A only for camera angle, crop, lens perspective, "
        "wall and floor geometry, subject reserved positions, and overall spatial layout. Use the generated background-depth fill "
        "reference to preserve empty-scene perspective and depth. Use the environment references for architecture, "
        "materials, lighting mood, color temperature, furniture, and scene details. Remove all people, faces, hands, bodies, "
        "and old clothing completely. Keep clean negative space where the two people will be composited back later."
    )
    negative = (
        "person, people, human body, face, hands, arms, legs, clothing, old outfit, silhouette remnants, person-shaped depth, "
        "old background artifacts, warped perspective, bad depth, pasted objects, blurry, low quality"
    )

    bg_plate = add_flux_inpaint_stage(
        wf,
        "stage2 background plate",
        scaled_a,
        person_feather,
        unet,
        clip,
        vae,
        [bg_depth_fill, env1, env2, env3],
        2860,
        -420,
        prompt,
        negative,
        0.86,
        24,
        91602024001,
        "vn_step1_step2/stage2_background_plate",
        width=1024,
        height=1536,
        cfg=1.2,
        guidance=3.5,
    )
    add_preview(wf, bg_plate, 0, 5520, 40, "background plate extra preview")

    data = {
        "id": "vn_step1_step2_background_plate_flux2",
        "revision": 0,
        "last_node_id": wf.next_node_id - 1,
        "last_link_id": wf.next_link_id - 1,
        "nodes": wf.nodes,
        "links": wf.links,
        "groups": wf.groups,
        "config": {},
        "extra": {"ds": {"scale": 0.42, "offset": [920, 560]}},
        "version": 0.4,
    }
    return data


def main():
    data = build()
    audit(data)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"nodes={len(data['nodes'])} links={len(data['links'])} groups={len(data['groups'])}")


if __name__ == "__main__":
    main()
