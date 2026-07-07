"""Local FLUX.2 Klein pipeline through a running ComfyUI server.

This runner uses the same job JSON, prompt builder, and reference ordering as
the hosted FLUX.2 Max and GPT Image 2 runners. It converts the saved ComfyUI UI
workflow to an API prompt at runtime, patches primitive/model/reference values,
prunes unused reference slots, queues the prompt, and copies the saved output
image into the tool's normal output directory.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from components import face_verify
from tool_lib.comfy_workflow import (
    audit_workflow_graph,
    convert_ui_workflow_to_api_prompt,
    history_output_images,
    post_json,
    read_history_image,
    wait_for_history,
)
from tool_lib.jobs import SUFFIX_BY_FORMAT, get_field, image_output_name, item_stem, normalize_items
from tool_lib.paths import REPO_ROOT, load_json, resolve_repo_path, unique_path
from tool_lib.prompting import build_prompt
from tool_lib.references import ensure_extensions, ref_summary, reference_log_entries, reference_paths

DEFAULT_WORKFLOW = Path("user/default/workflows/prod/lora_references/flux2_klein_lora_reference_image.json")
DEFAULT_LOG_DIR = Path("logs/flux2_klein_lora_references")
DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_UNET_NAME = r"flux2\flux-2-klein-9b-fp8.safetensors"
DEFAULT_CLIP_NAME = r"flux2\qwen_3_8b_fp8mixed.safetensors"
DEFAULT_VAE_NAME = r"flux2\full_encoder_small_decoder.safetensors"
DEFAULT_WEIGHT_DTYPE = "default"
DEFAULT_CLIP_TYPE = "flux2"
DEFAULT_DEVICE = "default"
DEFAULT_STEPS = 12
DEFAULT_GUIDANCE = 3.5
DEFAULT_SAMPLER_NAME = "euler"
DEFAULT_REFERENCE_METHOD = "index"
MAX_SEED = 2**63 - 1
COMFY_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

NODE_UNET = "1"
NODE_CLIP = "2"
NODE_VAE = "3"
NODE_PROMPT = "30"
NODE_REF_METHOD = "47"
NODE_GUIDANCE = "48"
NODE_WIDTH = "50"
NODE_HEIGHT = "51"
NODE_SCHEDULER = "52"
NODE_NOISE = "54"
NODE_SAMPLER = "55"
NODE_SAVE = "58"
REF_LOAD_IDS = [str(node_id) for node_id in range(10, 17)]
REF_ENCODE_IDS = [str(node_id) for node_id in range(20, 27)]
REF_LATENT_IDS = [str(node_id) for node_id in range(40, 47)]


def write_log(log_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / name
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log_path


def expect_node(prompt: dict[str, Any], node_id: str, class_type: str) -> dict[str, Any]:
    node = prompt.get(node_id)
    if not node:
        raise ValueError(f"Local workflow is missing expected node {node_id} ({class_type}).")
    if node.get("class_type") != class_type:
        raise ValueError(f"Local workflow node {node_id} must be {class_type}, got {node.get('class_type')}.")
    return node


def set_input(prompt: dict[str, Any], node_id: str, class_type: str, input_name: str, value: Any) -> None:
    node = expect_node(prompt, node_id, class_type)
    if input_name not in node.get("inputs", {}):
        raise ValueError(f"Node {node_id} ({class_type}) cannot accept input {input_name!r}.")
    node["inputs"][input_name] = value


def patch_load_image(node: dict[str, Any], path: Path) -> None:
    inputs = node.setdefault("inputs", {})
    inputs["image"] = str(path)
    if "clean_name" in inputs:
        inputs["clean_name"] = path.stem
    if "root_dir" in inputs:
        inputs["root_dir"] = str(path.parent)
    if "folder_name" in inputs:
        inputs["folder_name"] = ""
    if "filename_suffix" in inputs:
        inputs["filename_suffix"] = ""


def prune_reference_slots(prompt: dict[str, Any], refs: list[tuple[str, Path]]) -> None:
    ref_count = len(refs)
    for slot, (_, path) in enumerate(refs):
        load_node = expect_node(prompt, REF_LOAD_IDS[slot], "LoadImage")
        patch_load_image(load_node, path)

    for slot in range(ref_count, len(REF_LOAD_IDS)):
        prompt.pop(REF_LOAD_IDS[slot], None)
        prompt.pop(REF_ENCODE_IDS[slot], None)
        prompt.pop(REF_LATENT_IDS[slot], None)

    if ref_count == 0:
        prompt.pop(NODE_REF_METHOD, None)
        set_input(prompt, NODE_GUIDANCE, "FluxGuidance", "conditioning", [NODE_PROMPT, 0])
        return

    last_ref_id = REF_LATENT_IDS[ref_count - 1]
    expect_node(prompt, NODE_REF_METHOD, "FluxKontextMultiReferenceLatentMethod")
    prompt[NODE_REF_METHOD]["inputs"]["conditioning"] = [last_ref_id, 0]
    set_input(prompt, NODE_GUIDANCE, "FluxGuidance", "conditioning", [NODE_REF_METHOD, 0])


def patch_prompt(
    prompt: dict[str, Any],
    *,
    refs: list[tuple[str, Path]],
    prompt_text: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    guidance: float,
    sampler_name: str,
    reference_method: str,
    save_prefix: str,
    args: argparse.Namespace,
) -> None:
    prune_reference_slots(prompt, refs)
    set_input(prompt, NODE_UNET, "UNETLoader", "unet_name", args.klein_unet_name)
    set_input(prompt, NODE_UNET, "UNETLoader", "weight_dtype", args.klein_weight_dtype)
    set_input(prompt, NODE_CLIP, "CLIPLoader", "clip_name", args.klein_clip_name)
    set_input(prompt, NODE_CLIP, "CLIPLoader", "type", args.klein_clip_type)
    set_input(prompt, NODE_CLIP, "CLIPLoader", "device", args.klein_clip_device)
    set_input(prompt, NODE_VAE, "VAELoader", "vae_name", args.klein_vae_name)
    set_input(prompt, NODE_PROMPT, "CLIPTextEncode", "text", prompt_text)
    set_input(prompt, NODE_WIDTH, "PrimitiveInt", "value", width)
    set_input(prompt, NODE_HEIGHT, "PrimitiveInt", "value", height)
    set_input(prompt, NODE_SCHEDULER, "Flux2Scheduler", "steps", steps)
    set_input(prompt, NODE_NOISE, "RandomNoise", "noise_seed", seed)
    set_input(prompt, NODE_SAMPLER, "KSamplerSelect", "sampler_name", sampler_name)
    set_input(prompt, NODE_GUIDANCE, "FluxGuidance", "guidance", guidance)
    if refs:
        set_input(prompt, NODE_REF_METHOD, "FluxKontextMultiReferenceLatentMethod", "reference_latents_method", reference_method)
    set_input(prompt, NODE_SAVE, "SaveImage", "filename_prefix", save_prefix)


def save_image_bytes(data: bytes, target: Path, output_format: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        target.write_bytes(data)
        return
    try:
        from PIL import Image
        import io
    except ImportError:
        target.write_bytes(data)
        return
    with Image.open(io.BytesIO(data)) as image:
        image.save(target, format="JPEG" if output_format == "jpeg" else "WEBP")


def run(args: argparse.Namespace) -> int:
    workflow_path = resolve_repo_path(args.workflow)
    workflow = load_json(workflow_path)
    audit_workflow_graph(workflow)

    input_json = resolve_repo_path(args.input_json)
    items = normalize_items(load_json(input_json))
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("No prompt items found.")

    log_dir = resolve_repo_path(args.log_dir or DEFAULT_LOG_DIR)
    output_dir = resolve_repo_path(args.output_dir)
    suffix = SUFFIX_BY_FORMAT.get(args.output_format, ".png")
    total_runs = len(items) * args.repeat
    run_index = 0
    client_id = uuid.uuid4().hex

    for index, item in enumerate(items, start=1):
        base_stem = item_stem(item, index)
        refs = reference_paths(item, input_json.parent)
        ensure_extensions(refs, COMFY_IMAGE_EXTENSIONS, "ComfyUI LoadImage")
        prompt_text = build_prompt(item, refs)
        summary = ref_summary(refs)
        width = int(get_field(item, "width", default=args.width))
        height = int(get_field(item, "height", default=args.height))
        steps = int(get_field(item, "steps", default=args.steps))
        guidance = float(get_field(item, "guidance", default=args.guidance))
        sampler_name = str(get_field(item, "sampler_name", default=args.sampler_name))
        reference_method = str(get_field(item, "reference_method", default=args.reference_method))

        for repeat_index in range(1, args.repeat + 1):
            run_index += 1
            stem = f"{base_stem}_r{repeat_index:02d}" if args.repeat > 1 else base_stem
            seed_value = get_field(item, "seed", default=args.seed)
            seed = random.randint(0, MAX_SEED) if seed_value is None else int(seed_value)
            timestamp_s = int(time.time())
            stage_prefix = f"lora_data_generator/local_stage/{input_json.stem}_{run_index:05d}_{timestamp_s}"

            prompt = convert_ui_workflow_to_api_prompt(workflow)
            patch_prompt(
                prompt,
                refs=refs,
                prompt_text=prompt_text,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                guidance=guidance,
                sampler_name=sampler_name,
                reference_method=reference_method,
                save_prefix=stage_prefix,
                args=args,
            )

            log_payload: dict[str, Any] = {
                "run_index": run_index,
                "total_runs": total_runs,
                "item_index": index,
                "repeat_index": repeat_index,
                "repeat_count": args.repeat,
                "dry_run": args.dry_run,
                "output_stem": stem,
                "input_json_stem": input_json.stem,
                "output_dir": str(output_dir),
                "model": "flux-2-klein-9b-local",
                "workflow": str(workflow_path),
                "server": args.server,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "sampler_name": sampler_name,
                "reference_method": reference_method,
                "output_format": args.output_format,
                "references": reference_log_entries(refs),
                "prompt": prompt_text,
            }
            log_path = None
            if not args.no_log:
                log_path = write_log(log_dir, f"{run_index:05d}_{stem}.klein_request.json", log_payload)

            repeat_text = f" repeat {repeat_index}/{args.repeat}" if args.repeat > 1 else ""
            log_text = f"; log: {log_path}" if log_path else ""
            if args.dry_run:
                print(f"[dry-run {run_index}/{total_runs}] {base_stem}{repeat_text}: built local prompt with {summary} refs{log_text}")
                continue

            response = post_json(args.server, "/prompt", {"prompt": prompt, "client_id": client_id})
            prompt_id = response["prompt_id"]
            print(f"[{run_index}/{total_runs}] queued {base_stem}{repeat_text} ({summary} refs): {prompt_id}")
            history = wait_for_history(args.server, prompt_id, args.timeout)
            images = history_output_images(history)
            saved: list[Path] = []
            for image in images:
                data = read_history_image(args.server, image, REPO_ROOT / "output")
                name = image_output_name(input_json.stem, "klein", int(time.time()))
                target = unique_path(output_dir / f"{name}{suffix}")
                save_image_bytes(data, target, args.output_format)
                saved.append(target)

            names = ", ".join(path.name for path in saved) or "none"
            print(f"[{run_index}/{total_runs}] done {base_stem}{repeat_text}: saved {names}{log_text}")
            face_entries = face_verify.score_saved_images(refs, saved, getattr(args, "face_model_root", None))
            if log_path:
                log_payload.update(
                    {
                        "prompt_id": prompt_id,
                        "history_outputs": images,
                        "saved_images": [str(path) for path in saved],
                        "face_similarity": face_entries,
                    }
                )
                write_log(log_dir, log_path.name, log_payload)

    return 0
