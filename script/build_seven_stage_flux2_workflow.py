import json
from pathlib import Path


OUT = Path("user/default/workflows/dev/vn_seven_stage_flux2_local_depth_composite.json")


class Workflow:
    def __init__(self):
        self.nodes = []
        self.links = []
        self.groups = []
        self.next_node_id = 1
        self.next_link_id = 1
        self.order = 0

    def group(self, title, x, y, w, h, color="#334", font_size=24):
        gid = len(self.groups) + 1
        self.groups.append(
            {
                "id": gid,
                "title": title,
                "bounding": [x, y, w, h],
                "color": color,
                "font_size": font_size,
                "flags": {},
            }
        )
        return gid

    def node(self, node_type, x, y, inputs, outputs, widgets=None, title="", size=None, color=None, bgcolor=None, mode=0):
        nid = self.next_node_id
        self.next_node_id += 1
        node = {
            "id": nid,
            "type": node_type,
            "pos": [x, y],
            "size": size or [300, 90],
            "flags": {},
            "order": self.order,
            "mode": mode,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets or [],
        }
        self.order += 1
        if title:
            node["title"] = title
        if color:
            node["color"] = color
        if bgcolor:
            node["bgcolor"] = bgcolor
        self.nodes.append(node)
        return nid

    def link(self, origin, origin_slot, target, target_slot, link_type):
        lid = self.next_link_id
        self.next_link_id += 1
        self.links.append([lid, origin, origin_slot, target, target_slot, link_type])
        self.nodes_by_id[origin]["outputs"][origin_slot].setdefault("links", [])
        self.nodes_by_id[origin]["outputs"][origin_slot]["links"].append(lid)
        self.nodes_by_id[target]["inputs"][target_slot]["link"] = lid
        return lid

    @property
    def nodes_by_id(self):
        return {n["id"]: n for n in self.nodes}

    def out(self, name, typ, links=None):
        return {"localized_name": name, "name": name, "type": typ, "links": links or []}

    def inp(self, name, typ, widget=False, shape=None):
        d = {"localized_name": name, "name": name, "type": typ, "link": None}
        if widget:
            d["widget"] = {"name": name}
        if shape is not None:
            d["shape"] = shape
        return d


def add_load_image(wf, title, x, y, image="example.png"):
    return wf.node(
        "LoadImage",
        x,
        y,
        [wf.inp("image", "COMBO", True), wf.inp("upload", "IMAGEUPLOAD", True)],
        [wf.out("IMAGE", "IMAGE"), wf.out("MASK", "MASK")],
        [image, "image"],
        title=title,
        size=[300, 314],
        color="#232",
        bgcolor="#353",
    )


def add_preview(wf, image_node, image_slot, x, y, title):
    n = wf.node(
        "PreviewImage",
        x,
        y,
        [wf.inp("images", "IMAGE")],
        [],
        [],
        title=title,
        size=[420, 315],
        color="#223",
        bgcolor="#335",
    )
    wf.link(image_node, image_slot, n, 0, "IMAGE")
    return n


def add_save(wf, image_node, image_slot, x, y, prefix, title):
    n = wf.node(
        "SaveImage",
        x,
        y,
        [wf.inp("images", "IMAGE"), wf.inp("filename_prefix", "STRING", True)],
        [],
        [prefix],
        title=title,
        size=[360, 120],
    )
    wf.link(image_node, image_slot, n, 0, "IMAGE")
    return n


def add_mask_preview(wf, mask_node, mask_slot, x, y, title):
    mti = wf.node(
        "MaskToImage",
        x,
        y,
        [wf.inp("mask", "MASK")],
        [wf.out("IMAGE", "IMAGE")],
        [],
        title=title + " image",
        size=[220, 70],
        color="#223",
        bgcolor="#335",
    )
    wf.link(mask_node, mask_slot, mti, 0, "MASK")
    add_preview(wf, mti, 0, x + 260, y - 40, title)
    return mti


def add_image_scale(wf, image_node, x, y, width=1024, height=1536):
    n = wf.node(
        "ImageScale",
        x,
        y,
        [
            wf.inp("image", "IMAGE"),
            wf.inp("upscale_method", "COMBO", True),
            wf.inp("width", "INT", True),
            wf.inp("height", "INT", True),
            wf.inp("crop", "COMBO", True),
        ],
        [wf.out("IMAGE", "IMAGE")],
        ["lanczos", width, height, "center"],
        title="fixed final canvas",
        size=[315, 130],
        color="#223",
        bgcolor="#335",
    )
    wf.link(image_node, 0, n, 0, "IMAGE")
    return n


def add_person_mask(wf, image_node, x, y):
    det = wf.node(
        "UltralyticsDetectorProvider",
        x,
        y,
        [wf.inp("model_name", "COMBO", True)],
        [wf.out("BBOX_DETECTOR", "BBOX_DETECTOR"), wf.out("SEGM_DETECTOR", "SEGM_DETECTOR")],
        ["bbox/yolov8m.pt"],
        title="YOLO person bbox",
        size=[360, 58],
        color="#322",
        bgcolor="#533",
    )
    segs = wf.node(
        "BboxDetectorSEGS",
        x + 410,
        y,
        [
            wf.inp("bbox_detector", "BBOX_DETECTOR"),
            wf.inp("image", "IMAGE"),
            wf.inp("threshold", "FLOAT", True),
            wf.inp("dilation", "INT", True),
            wf.inp("crop_factor", "FLOAT", True),
            wf.inp("drop_size", "INT", True),
            wf.inp("labels", "STRING", True),
        ],
        [wf.out("SEGS", "SEGS")],
        [0.35, 0, 3.0, 10, "person"],
        title="detect all people",
        size=[360, 250],
        color="#322",
        bgcolor="#533",
    )
    sam = wf.node(
        "SAMLoader",
        x,
        y + 170,
        [wf.inp("model_name", "COMBO", True), wf.inp("device_mode", "COMBO", True)],
        [wf.out("SAM_MODEL", "SAM_MODEL")],
        ["sam_vit_b_01ec64.pth", "AUTO"],
        title="SAM loader",
        size=[360, 82],
        color="#322",
        bgcolor="#533",
    )
    mask = wf.node(
        "SAMDetectorCombined",
        x + 820,
        y,
        [
            wf.inp("sam_model", "SAM_MODEL"),
            wf.inp("segs", "SEGS"),
            wf.inp("image", "IMAGE"),
            wf.inp("detection_hint", "COMBO", True),
            wf.inp("dilation", "INT", True),
            wf.inp("threshold", "FLOAT", True),
            wf.inp("bbox_expansion", "INT", True),
            wf.inp("mask_hint_threshold", "FLOAT", True),
            wf.inp("mask_hint_use_negative", "COMBO", True),
        ],
        [wf.out("MASK", "MASK")],
        ["center-1", 0, 0.93, 10, 0.7, "False"],
        title="auto whole-person mask",
        size=[390, 325],
        color="#322",
        bgcolor="#533",
    )
    dilate = wf.node(
        "ImpactDilateMask",
        x + 1260,
        y,
        [wf.inp("mask", "MASK"), wf.inp("dilation", "INT", True)],
        [wf.out("MASK", "MASK")],
        [18],
        title="person mask expand",
        size=[260, 90],
        color="#223",
        bgcolor="#335",
    )
    blur = wf.node(
        "MaskBlur+",
        x + 1560,
        y,
        [wf.inp("mask", "MASK"), wf.inp("amount", "INT", True), wf.inp("device", "COMBO", True)],
        [wf.out("MASK", "MASK")],
        [36, "auto"],
        title="person mask feather",
        size=[280, 120],
        color="#223",
        bgcolor="#335",
    )
    wf.link(det, 0, segs, 0, "BBOX_DETECTOR")
    wf.link(image_node, 0, segs, 1, "IMAGE")
    wf.link(sam, 0, mask, 0, "SAM_MODEL")
    wf.link(segs, 0, mask, 1, "SEGS")
    wf.link(image_node, 0, mask, 2, "IMAGE")
    wf.link(mask, 0, dilate, 0, "MASK")
    wf.link(dilate, 0, blur, 0, "MASK")
    return mask, dilate, blur


def add_mask_invert(wf, mask_node, x, y, title):
    n = wf.node("InvertMask", x, y, [wf.inp("mask", "MASK")], [wf.out("MASK", "MASK")], [], title=title, size=[220, 70])
    wf.link(mask_node, 0, n, 0, "MASK")
    return n


def add_canny(wf, image_node, x, y):
    n = wf.node(
        "Canny",
        x,
        y,
        [wf.inp("image", "IMAGE"), wf.inp("low_threshold", "FLOAT", True), wf.inp("high_threshold", "FLOAT", True)],
        [wf.out("IMAGE", "IMAGE")],
        [0.25, 0.75],
        title="A structure canny",
        size=[300, 110],
        color="#223",
        bgcolor="#335",
    )
    wf.link(image_node, 0, n, 0, "IMAGE")
    return n


def add_da3_depth_fill(wf, image_node, person_mask_node, x, y, title_prefix="A"):
    wf.node(
        "MarkdownNote",
        x,
        y - 250,
        [],
        [],
        [
            "Download Depth Anything 3 model:\n\n"
            "https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/depth_anything_3_mono_large.safetensors\n\n"
            "Save to external model folder:\n\n"
            "C:/Users/Tony Xu/workspace/comfyui_models/geometry_estimation/depth_anything_3_mono_large.safetensors\n\n"
            "Restart ComfyUI after adding geometry_estimation to extra_model_paths.yaml or after downloading new model files."
        ],
        title=f"{title_prefix} DA3 download link + external path",
        size=[560, 220],
        color="#332",
        bgcolor="#554",
    )
    loader = wf.node(
        "LoadDA3Model",
        x,
        y,
        [wf.inp("model_name", "COMBO", True), wf.inp("weight_dtype", "COMBO", True)],
        [wf.out("DA3_MODEL", "DA3_MODEL")],
        ["depth_anything_3_mono_large.safetensors", "default"],
        title=f"{title_prefix} DA3 model loader - choose geometry_estimation model",
        size=[400, 140],
        color="#322",
        bgcolor="#533",
    )
    infer = wf.node(
        "DA3Inference",
        x + 460,
        y,
        [
            wf.inp("da3_model", "DA3_MODEL"),
            wf.inp("image", "IMAGE"),
            wf.inp("resolution", "INT", True),
            wf.inp("resize_method", "COMBO", True),
            wf.inp("mode", "COMFY_DYNAMICCOMBO_V3", True),
        ],
        [wf.out("da3_geometry", "DA3_GEOMETRY")],
        [1008, "upper_bound_resize", "mono"],
        title=f"{title_prefix} raw depth inference",
        size=[390, 130],
        color="#322",
        bgcolor="#533",
    )
    render = wf.node(
        "DA3Render",
        x + 920,
        y,
        [
            wf.inp("da3_geometry", "DA3_GEOMETRY"),
            wf.inp("output", "COMFY_DYNAMICCOMBO_V3", True),
            wf.inp("output.normalization", "COMBO", True),
            wf.inp("output.apply_sky_clip", "BOOLEAN", True),
        ],
        [wf.out("IMAGE", "IMAGE")],
        ["depth", "v2_style", False],
        title=f"{title_prefix} raw depth render",
        size=[380, 130],
        color="#223",
        bgcolor="#335",
    )
    blurred = wf.node(
        "ImageBlur",
        x + 920,
        y + 190,
        [
            wf.inp("image", "IMAGE"),
            wf.inp("blur_radius", "INT", True),
            wf.inp("sigma", "FLOAT", True),
        ],
        [wf.out("IMAGE", "IMAGE")],
        [31, 10.0],
        title=f"{title_prefix} blurred depth fill source",
        size=[300, 110],
        color="#223",
        bgcolor="#335",
    )
    fill = wf.node(
        "ImageCompositeMasked",
        x + 1300,
        y + 90,
        [
            wf.inp("destination", "IMAGE"),
            wf.inp("source", "IMAGE"),
            wf.inp("x", "INT", True),
            wf.inp("y", "INT", True),
            wf.inp("resize_source", "BOOLEAN", True),
            wf.inp("mask", "MASK", shape=7),
        ],
        [wf.out("IMAGE", "IMAGE")],
        [0, 0, False],
        title=f"{title_prefix} background-depth fill",
        size=[340, 150],
        color="#232",
        bgcolor="#353",
    )
    wf.link(loader, 0, infer, 0, "DA3_MODEL")
    wf.link(image_node, 0, infer, 1, "IMAGE")
    wf.link(infer, 0, render, 0, "DA3_GEOMETRY")
    wf.link(render, 0, blurred, 0, "IMAGE")
    wf.link(render, 0, fill, 0, "IMAGE")
    wf.link(blurred, 0, fill, 1, "IMAGE")
    wf.link(person_mask_node, 0, fill, 5, "MASK")
    return render, fill


def add_reference_chain(wf, conditioning_node, conditioning_slot, ref_image_nodes, vae_node, x, y, label):
    cond_node = conditioning_node
    cond_slot = conditioning_slot
    cur_y = y
    for idx, img_node in enumerate(ref_image_nodes, 1):
        enc = wf.node(
            "VAEEncode",
            x,
            cur_y,
            [wf.inp("pixels", "IMAGE"), wf.inp("vae", "VAE")],
            [wf.out("LATENT", "LATENT")],
            [],
            title=f"{label} ref {idx} encode",
            size=[260, 80],
        )
        ref = wf.node(
            "ReferenceLatent",
            x + 310,
            cur_y,
            [wf.inp("conditioning", "CONDITIONING"), wf.inp("latent", "LATENT", shape=7)],
            [wf.out("CONDITIONING", "CONDITIONING")],
            [],
            title=f"{label} ref {idx}",
            size=[280, 70],
            color="#232",
            bgcolor="#353",
        )
        wf.link(img_node, 0, enc, 0, "IMAGE")
        wf.link(vae_node, 0, enc, 1, "VAE")
        wf.link(cond_node, cond_slot, ref, 0, "CONDITIONING")
        wf.link(enc, 0, ref, 1, "LATENT")
        cond_node, cond_slot = ref, 0
        cur_y += 105
    method = wf.node(
        "FluxKontextMultiReferenceLatentMethod",
        x + 640,
        y,
        [wf.inp("conditioning", "CONDITIONING"), wf.inp("reference_latents_method", "COMBO", True)],
        [wf.out("CONDITIONING", "CONDITIONING")],
        ["index"],
        title=f"{label} multi-reference method",
        size=[320, 70],
        color="#232",
        bgcolor="#353",
    )
    wf.link(cond_node, cond_slot, method, 0, "CONDITIONING")
    return method


def add_flux_inpaint_stage(
    wf,
    name,
    base_image_node,
    mask_node,
    model_node,
    clip_node,
    vae_node,
    refs,
    x,
    y,
    prompt,
    negative,
    denoise,
    steps,
    seed,
    save_prefix,
    width=1024,
    height=1536,
    cfg=1.2,
    guidance=3.5,
    base_image_slot=0,
    mask_slot=0,
):
    pos = wf.node(
        "CLIPTextEncode",
        x,
        y,
        [wf.inp("clip", "CLIP"), wf.inp("text", "STRING", True)],
        [wf.out("CONDITIONING", "CONDITIONING")],
        [prompt],
        title=f"{name} positive",
        size=[520, 260],
        color="#232",
        bgcolor="#353",
    )
    neg = wf.node(
        "CLIPTextEncode",
        x,
        y + 300,
        [wf.inp("clip", "CLIP"), wf.inp("text", "STRING", True)],
        [wf.out("CONDITIONING", "CONDITIONING")],
        [negative],
        title=f"{name} negative",
        size=[520, 170],
        color="#322",
        bgcolor="#533",
    )
    wf.link(clip_node, 0, pos, 0, "CLIP")
    wf.link(clip_node, 0, neg, 0, "CLIP")

    cond = add_reference_chain(wf, pos, 0, refs, vae_node, x + 590, y, name)
    guide = wf.node(
        "FluxGuidance",
        x + 1570,
        y,
        [wf.inp("conditioning", "CONDITIONING"), wf.inp("guidance", "FLOAT", True)],
        [wf.out("CONDITIONING", "CONDITIONING")],
        [guidance],
        title=f"{name} guidance",
        size=[300, 70],
    )
    guider = wf.node(
        "CFGGuider",
        x + 1900,
        y,
        [wf.inp("model", "MODEL"), wf.inp("positive", "CONDITIONING"), wf.inp("negative", "CONDITIONING"), wf.inp("cfg", "FLOAT", True)],
        [wf.out("GUIDER", "GUIDER")],
        [cfg],
        title=f"{name} guider",
        size=[280, 130],
    )
    base_latent = wf.node(
        "VAEEncode",
        x + 1570,
        y + 180,
        [wf.inp("pixels", "IMAGE"), wf.inp("vae", "VAE")],
        [wf.out("LATENT", "LATENT")],
        [],
        title=f"{name} encode base",
        size=[260, 80],
    )
    masked = None
    if mask_node is not None:
        masked = wf.node(
            "SetLatentNoiseMask",
            x + 1900,
            y + 180,
            [wf.inp("samples", "LATENT"), wf.inp("mask", "MASK")],
            [wf.out("LATENT", "LATENT")],
            [],
            title=f"{name} noise mask",
            size=[300, 90],
        )
    noise = wf.node(
        "RandomNoise",
        x + 1570,
        y + 340,
        [wf.inp("noise_seed", "INT", True)],
        [wf.out("NOISE", "NOISE")],
        [seed, "randomize"],
        title=f"{name} seed",
        size=[300, 82],
    )
    sampler = wf.node(
        "KSamplerSelect",
        x + 1900,
        y + 340,
        [wf.inp("sampler_name", "COMBO", True)],
        [wf.out("SAMPLER", "SAMPLER")],
        ["euler"],
        title=f"{name} sampler",
        size=[300, 58],
    )
    sched = wf.node(
        "Flux2Scheduler",
        x + 2230,
        y + 340,
        [wf.inp("steps", "INT", True), wf.inp("width", "INT", True), wf.inp("height", "INT", True)],
        [wf.out("SIGMAS", "SIGMAS")],
        [steps, width, height],
        title=f"{name} scheduler",
        size=[300, 106],
    )
    split = wf.node(
        "SplitSigmasDenoise",
        x + 2560,
        y + 340,
        [wf.inp("sigmas", "SIGMAS"), wf.inp("denoise", "FLOAT", True)],
        [wf.out("high_sigmas", "SIGMAS"), wf.out("low_sigmas", "SIGMAS")],
        [denoise],
        title=f"{name} denoise {denoise}",
        size=[300, 82],
    )
    sample = wf.node(
        "SamplerCustomAdvanced",
        x + 2230,
        y + 520,
        [
            wf.inp("noise", "NOISE"),
            wf.inp("guider", "GUIDER"),
            wf.inp("sampler", "SAMPLER"),
            wf.inp("sigmas", "SIGMAS"),
            wf.inp("latent_image", "LATENT"),
        ],
        [wf.out("output", "LATENT"), wf.out("denoised_output", "LATENT")],
        [],
        title=f"{name} sample",
        size=[340, 130],
    )
    decode = wf.node(
        "VAEDecode",
        x + 2630,
        y + 530,
        [wf.inp("samples", "LATENT"), wf.inp("vae", "VAE")],
        [wf.out("IMAGE", "IMAGE")],
        [],
        title=f"{name} decoded",
        size=[190, 70],
    )
    wf.link(cond, 0, guide, 0, "CONDITIONING")
    wf.link(model_node, 0, guider, 0, "MODEL")
    wf.link(guide, 0, guider, 1, "CONDITIONING")
    wf.link(neg, 0, guider, 2, "CONDITIONING")
    wf.link(base_image_node, base_image_slot, base_latent, 0, "IMAGE")
    wf.link(vae_node, 0, base_latent, 1, "VAE")
    if masked is not None:
        wf.link(base_latent, 0, masked, 0, "LATENT")
        wf.link(mask_node, mask_slot, masked, 1, "MASK")
    wf.link(sched, 0, split, 0, "SIGMAS")
    wf.link(noise, 0, sample, 0, "NOISE")
    wf.link(guider, 0, sample, 1, "GUIDER")
    wf.link(sampler, 0, sample, 2, "SAMPLER")
    wf.link(split, 1, sample, 3, "SIGMAS")
    if masked is not None:
        wf.link(masked, 0, sample, 4, "LATENT")
    else:
        wf.link(base_latent, 0, sample, 4, "LATENT")
    wf.link(sample, 0, decode, 0, "LATENT")
    wf.link(vae_node, 0, decode, 1, "VAE")
    add_preview(wf, decode, 0, x + 2880, y + 360, f"{name} preview")
    add_save(wf, decode, 0, x + 2880, y + 710, save_prefix, f"{name} save")
    return decode


def add_flux_img2img_stage(
    wf,
    name,
    base_image_node,
    model_node,
    clip_node,
    vae_node,
    refs,
    x,
    y,
    prompt,
    negative,
    denoise,
    steps,
    seed,
    save_prefix,
    width=1024,
    height=1536,
):
    return add_flux_inpaint_stage(
        wf,
        name,
        base_image_node,
        None,
        model_node,
        clip_node,
        vae_node,
        refs,
        x,
        y,
        prompt,
        negative,
        denoise,
        steps,
        seed,
        save_prefix,
        width,
        height,
        cfg=1.15,
        guidance=3.0,
    )


def add_composite_people(wf, bg_node, people_node, mask_node, x, y):
    comp = wf.node(
        "ImageCompositeMasked",
        x,
        y,
        [
            wf.inp("destination", "IMAGE"),
            wf.inp("source", "IMAGE"),
            wf.inp("x", "INT", True),
            wf.inp("y", "INT", True),
            wf.inp("resize_source", "BOOLEAN", True),
            wf.inp("mask", "MASK", shape=7),
        ],
        [wf.out("IMAGE", "IMAGE")],
        [0, 0, False],
        title="temporary paste A people on new background",
        size=[340, 150],
        color="#232",
        bgcolor="#353",
    )
    wf.link(bg_node, 0, comp, 0, "IMAGE")
    wf.link(people_node, 0, comp, 1, "IMAGE")
    wf.link(mask_node, 0, comp, 5, "MASK")
    return comp


def add_ring_mask(wf, inner_mask, expanded_mask, x, y):
    sub = wf.node(
        "SubtractMask",
        x,
        y,
        [wf.inp("mask1", "MASK"), wf.inp("mask2", "MASK")],
        [wf.out("MASK", "MASK")],
        [],
        title="edge ring = expanded - person",
        size=[230, 80],
    )
    blur = wf.node(
        "MaskBlur+",
        x + 270,
        y,
        [wf.inp("mask", "MASK"), wf.inp("amount", "INT", True), wf.inp("device", "COMBO", True)],
        [wf.out("MASK", "MASK")],
        [28, "auto"],
        title="edge ring feather",
        size=[280, 120],
    )
    wf.link(expanded_mask, 0, sub, 0, "MASK")
    wf.link(inner_mask, 0, sub, 1, "MASK")
    wf.link(sub, 0, blur, 0, "MASK")
    return blur


def add_mask_load_as_mask(wf, title, x, y, image="example.png"):
    load = add_load_image(wf, title, x, y, image)
    return load


def build():
    wf = Workflow()
    wf.group("Models and reusable loaders", -1900, -680, 520, 620, "#433")
    wf.group("Reference images and manual mask override inputs", -1900, 80, 1320, 1540, "#343")
    wf.group("1. Preprocess A: masks and background-depth fill", -520, -680, 2700, 1540, "#334")
    wf.group("2. Generate empty background plate from A depth/layout", 2100, -680, 3740, 1020, "#443")
    wf.group("3. Composite A people back + edge blend", 2100, 520, 3740, 1080, "#334")
    wf.group("4. Clothing and body repaint", 2100, 1780, 3740, 1080, "#343")
    wf.group("5. Cropped face and hair refinement", 2100, 3040, 3740, 1160, "#433")
    wf.group("6. Cropped hands/contact repair", 2100, 4400, 3740, 1160, "#334")
    wf.group("7. Full-image low-denoise harmonize", 2100, 5760, 3740, 1020, "#443")

    unet = wf.node(
        "UNETLoader",
        -1860,
        -610,
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
        -1860,
        -480,
        [wf.inp("clip_name", "COMBO", True), wf.inp("type", "COMBO", True), wf.inp("device", "COMBO", True)],
        [wf.out("CLIP", "CLIP")],
        ["flux2\\qwen_3_8b_fp8mixed.safetensors", "flux2", "default"],
        title="FLUX.2 text encoder",
        size=[340, 106],
        color="#322",
        bgcolor="#533",
    )
    vae = wf.node(
        "VAELoader",
        -1860,
        -330,
        [wf.inp("vae_name", "COMBO", True)],
        [wf.out("VAE", "VAE")],
        ["flux2\\full_encoder_small_decoder.safetensors"],
        title="FLUX.2 VAE",
        size=[340, 58],
        color="#322",
        bgcolor="#533",
    )
    wf.node(
        "MarkdownNote",
        -1860,
        -230,
        [],
        [],
        [
            "Depth note: load an A-derived background-depth plate below. Generate it from A by estimating depth, masking people, and inpainting/filling the masked depth region. Raw A depth is useful as a secondary structure reference, but the background plate should not contain person-shaped depth silhouettes."
        ],
        title="Depth strategy",
        size=[430, 180],
        color="#332",
        bgcolor="#554",
    )

    a = add_load_image(wf, "A original composition image", -1860, 140, "example.png")
    pose = add_load_image(wf, "A DWPose/OpenPose reference optional", -840, 140, "example.png")
    bg1 = add_load_image(wf, "Environment reference #1 - replace before queue", -1860, 520, "example.png")
    bg2 = add_load_image(wf, "Environment reference #2 - replace before queue", -1520, 520, "example.png")
    bg3 = add_load_image(wf, "Environment reference #3 - replace before queue", -1180, 520, "example.png")
    outfit1 = add_load_image(wf, "Outfit/body reference #1", -1860, 900, "example.png")
    outfit2 = add_load_image(wf, "Outfit/body reference #2", -1520, 900, "example.png")
    outfit3 = add_load_image(wf, "Outfit/body reference #3", -1180, 900, "example.png")
    face1 = add_load_image(wf, "Face reference #1", -1860, 1280, "example.png")
    face2 = add_load_image(wf, "Face reference #2", -1520, 1280, "example.png")
    face3 = add_load_image(wf, "Face reference #3", -1180, 1280, "example.png")
    manual_face = add_mask_load_as_mask(wf, "Manual face/hair mask override", -760, 520, "example.png")
    manual_outfit = add_mask_load_as_mask(wf, "Manual outfit/body mask override", -760, 900, "example.png")
    manual_contact = add_mask_load_as_mask(wf, "Manual hands/contact mask", -760, 1280, "example.png")

    scaled_a = add_image_scale(wf, a, -480, -520)
    add_preview(wf, scaled_a, 0, -120, -610, "A fixed canvas preview")
    auto_person, person_expanded, person_feather = add_person_mask(wf, scaled_a, -480, -260)
    bg_mask = add_mask_invert(wf, person_feather, 1340, 220, "background mask preview helper")
    raw_depth, bg_depth_fill = add_da3_depth_fill(wf, scaled_a, person_feather, -480, 500, "A")
    add_preview(wf, raw_depth, 0, 1360, 470, "raw depth debug only")
    add_preview(wf, bg_depth_fill, 0, 1360, 820, "STEP 2 STRUCTURE REF - background-depth fill")
    add_mask_preview(wf, auto_person, 0, 1040, -300, "auto person mask raw")
    add_mask_preview(wf, person_feather, 0, 1040, 120, "person feather mask")
    add_mask_preview(wf, bg_mask, 0, 1040, 500, "background mask")

    bg_prompt = (
        "Generate an empty background plate with no people. Use the original image only for camera angle, crop, lens perspective, wall/floor depth, subject reserved positions, and overall spatial layout. "
        "Use the generated background-depth fill as the structure guide for empty-scene perspective and depth. Use the environment reference images for architecture, materials, lighting mood, color temperature, furniture, and background detail. "
        "Remove all people and old clothing/faces completely. Keep perspective consistent and leave clean negative space where the two people will be composited back later."
    )
    neg_bg = "people, person, human body, face, hands, clothes, old background artifacts, warped perspective, bad depth, pasted objects, low quality, blurry"
    bg_plate = add_flux_inpaint_stage(
        wf,
        "stage2 background plate",
        scaled_a,
        person_feather,
        unet,
        clip,
        vae,
        [bg_depth_fill, bg1, bg2, bg3],
        2140,
        -590,
        bg_prompt,
        neg_bg,
        0.86,
        24,
        91602024001,
        "vn_seven_stage_flux2/stage2_background_plate",
    )

    comp = add_composite_people(wf, bg_plate, scaled_a, auto_person, 2140, 650)
    add_preview(wf, comp, 0, 2540, 580, "temporary composite preview")
    edge_ring = add_ring_mask(wf, auto_person, person_expanded, 2140, 1040)
    add_mask_preview(wf, edge_ring, 0, 2720, 1030, "edge blend ring mask")
    edge_prompt = (
        "Blend only the pasted people edges into the new background. Preserve exact body pose, hand placement, facial expression, scale, overlap, and spatial relationship. "
        "Only adjust edge lighting, hair boundary, clothing boundary, color spill, contact shadows, and ambient occlusion. Do not change faces, hands, clothing design, body shape, or pose."
    )
    edge_blend = add_flux_inpaint_stage(
        wf,
        "stage3 edge blend",
        comp,
        edge_ring,
        unet,
        clip,
        vae,
        [scaled_a, bg_plate, bg_depth_fill],
        2140,
        870,
        edge_prompt,
        "changed pose, changed faces, changed hands, changed clothing, new objects, visible seam, halo, sticker edge, blurry edge",
        0.32,
        16,
        91602024002,
        "vn_seven_stage_flux2/stage3_people_edge_blend",
    )

    face_protect = wf.node(
        "SolidMask",
        2140,
        1840,
        [wf.inp("value", "FLOAT", True), wf.inp("width", "INT", True), wf.inp("height", "INT", True)],
        [wf.out("MASK", "MASK")],
        [0.0, 1024, 1536],
        title="auto face/hair mask placeholder; replace with FaceParsing branch if needed",
        size=[310, 106],
    )
    rough_outfit = wf.node(
        "MaskComposite",
        2500,
        1840,
        [wf.inp("destination", "MASK"), wf.inp("source", "MASK"), wf.inp("x", "INT", True), wf.inp("y", "INT", True), wf.inp("operation", "COMBO", True)],
        [wf.out("MASK", "MASK")],
        [0, 0, "add"],
        title="outfit mask = auto person + manual override",
        size=[310, 120],
    )
    wf.link(person_feather, 0, rough_outfit, 0, "MASK")
    wf.link(manual_outfit, 1, rough_outfit, 1, "MASK")
    subtract_face = wf.node(
        "SubtractMask",
        2860,
        1840,
        [wf.inp("mask1", "MASK"), wf.inp("mask2", "MASK")],
        [wf.out("MASK", "MASK")],
        [],
        title="protect face/hair from clothing pass",
        size=[260, 80],
    )
    wf.link(rough_outfit, 0, subtract_face, 0, "MASK")
    wf.link(manual_face, 1, subtract_face, 1, "MASK")
    subtract_contact = wf.node(
        "SubtractMask",
        3160,
        1840,
        [wf.inp("mask1", "MASK"), wf.inp("mask2", "MASK")],
        [wf.out("MASK", "MASK")],
        [],
        title="protect hands/contact from clothing pass",
        size=[260, 80],
    )
    wf.link(subtract_face, 0, subtract_contact, 0, "MASK")
    wf.link(manual_contact, 1, subtract_contact, 1, "MASK")
    outfit_blur = wf.node(
        "MaskBlur+",
        3460,
        1840,
        [wf.inp("mask", "MASK"), wf.inp("amount", "INT", True), wf.inp("device", "COMBO", True)],
        [wf.out("MASK", "MASK")],
        [32, "auto"],
        title="outfit/body feather mask",
        size=[280, 120],
    )
    wf.link(subtract_contact, 0, outfit_blur, 0, "MASK")
    add_mask_preview(wf, outfit_blur, 0, 3780, 1840, "outfit/body repaint mask")
    outfit_prompt = (
        "Repaint only the masked clothing and body-shape area. Preserve the same camera angle, body pose, hand positions, face, hair, background, interaction, and person spacing. "
        "Use outfit/body references for garment silhouette, fabric, color, seams, folds, fit, and body proportions. Keep contact points and hands untouched. Match lighting and shadows from the new background."
    )
    outfit_stage = add_flux_inpaint_stage(
        wf,
        "stage4 clothing body",
        edge_blend,
        outfit_blur,
        unet,
        clip,
        vae,
        [scaled_a, bg_depth_fill, pose, outfit1, outfit2, outfit3],
        2140,
        2100,
        outfit_prompt,
        "changed face, changed hands, broken fingers, changed pose, changed background, wrong perspective, visible mask edge, pasted clothing, bad fabric, blurry",
        0.64,
        20,
        91602024003,
        "vn_seven_stage_flux2/stage4_clothing_body",
    )

    face_mask = wf.node(
        "MaskComposite",
        2140,
        3160,
        [wf.inp("destination", "MASK"), wf.inp("source", "MASK"), wf.inp("x", "INT", True), wf.inp("y", "INT", True), wf.inp("operation", "COMBO", True)],
        [wf.out("MASK", "MASK")],
        [0, 0, "add"],
        title="face/hair mask from manual override",
        size=[320, 120],
    )
    wf.link(face_protect, 0, face_mask, 0, "MASK")
    wf.link(manual_face, 1, face_mask, 1, "MASK")
    face_bbox = wf.node(
        "MaskBoundingBox+",
        2500,
        3160,
        [wf.inp("mask", "MASK"), wf.inp("image_optional", "IMAGE", shape=7), wf.inp("padding", "INT", True), wf.inp("blur", "INT", True)],
        [
            wf.out("MASK", "MASK"),
            wf.out("IMAGE", "IMAGE"),
            wf.out("x", "INT"),
            wf.out("y", "INT"),
            wf.out("width", "INT"),
            wf.out("height", "INT"),
        ],
        [180, 18],
        title="crop face/hair with shoulder context",
        size=[330, 190],
    )
    wf.link(face_mask, 0, face_bbox, 0, "MASK")
    wf.link(outfit_stage, 0, face_bbox, 1, "IMAGE")
    add_preview(wf, face_bbox, 1, 2860, 3120, "face crop preview")
    face_prompt = (
        "Refine only the cropped face and hair region. Use the face references for identity, facial structure, skin texture, eyes, nose, lips, hairline, and hair style. "
        "Keep the same facial expression, gaze direction, head tilt, camera angle, body pose, outfit, and background. Blend skin and hair edges naturally into the crop context."
    )
    face_crop_stage = add_flux_inpaint_stage(
        wf,
        "stage5 face crop",
        face_bbox,
        face_bbox,
        unet,
        clip,
        vae,
        [face1, face2, face3, scaled_a],
        2140,
        3420,
        face_prompt,
        "changed expression, changed gaze, changed head angle, changed outfit, changed pose, distorted eyes, asymmetric face, bad teeth, harsh seam, blurry face",
        0.46,
        18,
        91602024004,
        "vn_seven_stage_flux2/stage5_face_crop_debug",
        width=768,
        height=768,
        base_image_slot=1,
        mask_slot=0,
    )
    paste_face = wf.node(
        "ImageCompositeMasked",
        5100,
        3300,
        [
            wf.inp("destination", "IMAGE"),
            wf.inp("source", "IMAGE"),
            wf.inp("x", "INT", True),
            wf.inp("y", "INT", True),
            wf.inp("resize_source", "BOOLEAN", True),
            wf.inp("mask", "MASK", shape=7),
        ],
        [wf.out("IMAGE", "IMAGE")],
        [0, 0, False],
        title="paste refined face crop back",
        size=[340, 150],
        color="#232",
        bgcolor="#353",
    )
    wf.link(outfit_stage, 0, paste_face, 0, "IMAGE")
    wf.link(face_crop_stage, 0, paste_face, 1, "IMAGE")
    wf.link(face_bbox, 2, paste_face, 2, "INT")
    wf.link(face_bbox, 3, paste_face, 3, "INT")
    wf.link(face_bbox, 0, paste_face, 5, "MASK")
    add_preview(wf, paste_face, 0, 5100, 3520, "face pasted full image")
    add_save(wf, paste_face, 0, 5100, 3880, "vn_seven_stage_flux2/stage5_face_pasted", "stage5 face paste save")

    contact_bbox = wf.node(
        "MaskBoundingBox+",
        2500,
        4520,
        [wf.inp("mask", "MASK"), wf.inp("image_optional", "IMAGE", shape=7), wf.inp("padding", "INT", True), wf.inp("blur", "INT", True)],
        [
            wf.out("MASK", "MASK"),
            wf.out("IMAGE", "IMAGE"),
            wf.out("x", "INT"),
            wf.out("y", "INT"),
            wf.out("width", "INT"),
            wf.out("height", "INT"),
        ],
        [160, 12],
        title="crop hands/contact/feet shadow area",
        size=[330, 190],
    )
    wf.link(manual_contact, 1, contact_bbox, 0, "MASK")
    wf.link(paste_face, 0, contact_bbox, 1, "IMAGE")
    add_preview(wf, contact_bbox, 1, 2860, 4480, "contact crop preview")
    hand_prompt = (
        "Repair only hands, shoulder contact, feet contact, and contact shadows inside the crop. Preserve exact hand geometry and placement from the original structure reference. "
        "Adjust skin tone, edge light, ambient occlusion, pressure shadow, and local texture so the contact belongs naturally in the new background. Do not move arms, hands, fingers, shoulders, faces, or bodies."
    )
    hand_crop_stage = add_flux_inpaint_stage(
        wf,
        "stage6 hands contact crop",
        contact_bbox,
        contact_bbox,
        unet,
        clip,
        vae,
        [scaled_a, bg_depth_fill, pose, paste_face],
        2140,
        4780,
        hand_prompt,
        "changed hand placement, extra fingers, missing fingers, broken hands, changed pose, changed face, changed clothing, harsh seam, blurry contact shadow",
        0.34,
        16,
        91602024005,
        "vn_seven_stage_flux2/stage6_hands_contact_crop_debug",
        width=768,
        height=768,
        base_image_slot=1,
        mask_slot=0,
    )
    paste_hand = wf.node(
        "ImageCompositeMasked",
        5100,
        4660,
        [
            wf.inp("destination", "IMAGE"),
            wf.inp("source", "IMAGE"),
            wf.inp("x", "INT", True),
            wf.inp("y", "INT", True),
            wf.inp("resize_source", "BOOLEAN", True),
            wf.inp("mask", "MASK", shape=7),
        ],
        [wf.out("IMAGE", "IMAGE")],
        [0, 0, False],
        title="paste repaired hands/contact crop back",
        size=[340, 150],
        color="#232",
        bgcolor="#353",
    )
    wf.link(paste_face, 0, paste_hand, 0, "IMAGE")
    wf.link(hand_crop_stage, 0, paste_hand, 1, "IMAGE")
    wf.link(contact_bbox, 2, paste_hand, 2, "INT")
    wf.link(contact_bbox, 3, paste_hand, 3, "INT")
    wf.link(contact_bbox, 0, paste_hand, 5, "MASK")
    add_preview(wf, paste_hand, 0, 5100, 4880, "hands/contact pasted full image")
    add_save(wf, paste_hand, 0, 5100, 5240, "vn_seven_stage_flux2/stage6_hands_contact_pasted", "stage6 save")

    harmonize_prompt = (
        "Low-denoise full image harmonization only. Unify lighting direction, color temperature, exposure, skin reflections, clothing shadows, background bounce light, ambient occlusion, contact shadows, grain, and lens rendering. "
        "Do not change composition, pose, identities, outfits, background objects, faces, hands, or spatial relationships."
    )
    final = add_flux_img2img_stage(
        wf,
        "stage7 harmonize",
        paste_hand,
        unet,
        clip,
        vae,
        [paste_hand, bg_plate, scaled_a, bg_depth_fill],
        2140,
        5860,
        harmonize_prompt,
        "changed pose, changed face, changed hands, changed clothing, changed background, warped geometry, visible seam, over-smoothed skin, low quality, blurry",
        0.20,
        14,
        91602024006,
        "vn_seven_stage_flux2/final_harmonized",
    )
    add_preview(wf, final, 0, 5100, 6060, "FINAL harmonized preview")

    data = {
        "id": "vn_seven_stage_flux2_local_depth_composite",
        "revision": 0,
        "last_node_id": wf.next_node_id - 1,
        "last_link_id": wf.next_link_id - 1,
        "nodes": wf.nodes,
        "links": wf.links,
        "groups": wf.groups,
        "config": {},
        "extra": {"ds": {"scale": 0.32, "offset": [1200, 400]}},
        "version": 0.4,
    }
    return data


def audit(data):
    nodes = {n["id"]: n for n in data["nodes"]}
    link_ids = [link[0] for link in data["links"]]
    if len(link_ids) != len(set(link_ids)):
        raise ValueError("duplicate link IDs detected")
    links_by_id = {link[0]: link for link in data["links"]}
    input_refs = set()
    output_refs = set()
    for link in data["links"]:
        if len(link) != 6:
            raise ValueError(f"bad link array: {link}")
        lid, src, src_slot, dst, dst_slot, typ = link
        if src not in nodes or dst not in nodes:
            raise ValueError(f"dangling node ref in link {lid}")
        if src_slot >= len(nodes[src].get("outputs", [])):
            raise ValueError(f"bad source slot in link {lid}")
        if dst_slot >= len(nodes[dst].get("inputs", [])):
            raise ValueError(f"bad target slot in link {lid}")
        src_type = nodes[src]["outputs"][src_slot].get("type")
        dst_type = nodes[dst]["inputs"][dst_slot].get("type")
        if src_type != typ or dst_type != typ:
            raise ValueError(f"type mismatch in link {lid}: {src_type}->{dst_type} declared {typ}")
        input_refs.add((dst, dst_slot, lid))
        output_refs.add((src, src_slot, lid))

    for node in data["nodes"]:
        for idx, inp in enumerate(node.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None:
                if lid not in links_by_id:
                    raise ValueError(f"input dangling link {lid} on node {node['id']}")
                link = links_by_id[lid]
                if link[3] != node["id"] or link[4] != idx:
                    raise ValueError(f"input link {lid} target mismatch on node {node['id']}")
        for idx, out in enumerate(node.get("outputs", [])):
            for lid in out.get("links") or []:
                if lid not in links_by_id:
                    raise ValueError(f"output dangling link {lid} on node {node['id']}")
                link = links_by_id[lid]
                if link[1] != node["id"] or link[2] != idx:
                    raise ValueError(f"output link {lid} source mismatch on node {node['id']}")


def main():
    data = build()
    audit(data)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"nodes={len(data['nodes'])} links={len(data['links'])} groups={len(data['groups'])}")


if __name__ == "__main__":
    main()
