"""Prompt construction shared by both pipelines.

build_prompt() is the single source of truth for the final prompt text. Both
the Flux.2 Max and GPT Image 2 pipelines send this exact string, so generation
comparisons are apples-to-apples: any change here affects both models equally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tool_lib.jobs import get_field, normalize_text


def reference_order_block(refs: list[tuple[str, Path]]) -> str:
    if not refs:
        return "Attached reference image order:\nNo reference images are attached for this run."

    dressing_indices = [index for index, (kind, _) in enumerate(refs, start=1) if kind == "dressing"]
    character_indices = [index for index, (kind, _) in enumerate(refs, start=1) if kind == "character"]
    lines = ["Attached reference image order:"]
    for index, (kind, path) in enumerate(refs, start=1):
        role = "dressing/outfit reference" if kind == "dressing" else "character face/body identity reference"
        lines.append(f"- Reference image {index}: {role}. Source file: {path.name}.")

    if dressing_indices:
        values = ", ".join(f"Reference image {index}" for index in dressing_indices)
        lines.append(f"Use {values} only for outfit, garment structure, fabric, color, styling, and dressing details.")
    if character_indices:
        values = ", ".join(f"Reference image {index}" for index in character_indices)
        lines.append(f"Use {values} only for the woman's face identity, body proportions, skin tone, age impression, and overall presence.")
    if dressing_indices and character_indices:
        lines.append(
            "Do not take the face, body, age, ethnicity, or hairstyle from the dressing references. "
            "Do not let outfit details override the character identity references."
        )
    return "\n".join(lines)


def build_prompt(item: dict[str, Any], refs: list[tuple[str, Path]]) -> str:
    direct_prompt = normalize_text(get_field(item, "prompt", "positive_prompt"))
    order_block = reference_order_block(refs)
    if direct_prompt:
        return "\n\n".join([order_block, direct_prompt])

    task = normalize_text(
        get_field(
            item,
            "task_output_goal",
            "task",
            "output_goal",
            default="Create one natural, photorealistic character reference photograph for synthetic LoRA training.",
        )
    )
    reference_priority = normalize_text(
        get_field(
            item,
            "reference_priority",
            default=(
                "- Face and identity: follow the character reference images exactly.\n"
                "- Body shape and proportions: follow the character reference images exactly.\n"
                "- Outfit: follow the dressing reference images or the outfit description below.\n"
                "- If there is conflict between the written prompt and the reference images, keep the character identity from the reference images."
            ),
        )
    )
    identity = normalize_text(
        get_field(
            item,
            "identity_lock",
            "character_identity",
            "identity",
            default=(
                "She has the same face, facial proportions, eye shape, nose shape, lips, jawline, "
                "cheek structure, skin tone, and natural facial texture as in the reference images. "
                "Her hair is long dark brown, softly wavy, with loose natural curls. Keep her body "
                "proportions consistent with the reference images, including shoulder width, waist, "
                "bust, hip proportions, and overall height impression. Her appearance should remain "
                "realistic, elegant, and natural."
            ),
        )
    )
    outfit = normalize_text(get_field(item, "outfit_block", "outfit", default="Follow the dressing reference images closely."))
    shot_type = normalize_text(get_field(item, "shot_type", "shot", "framing", default="Half-body portrait"))
    camera_view = normalize_text(get_field(item, "camera_view", "angle", default="front 3/4 view"))
    pose = normalize_text(get_field(item, "pose", "action", default="standing naturally"))
    expression = normalize_text(get_field(item, "expression", default="neutral, relaxed expression"))
    environment = normalize_text(get_field(item, "environment_block", "environment", "background", default="Simple clean studio light-colored background."))
    lighting = normalize_text(
        get_field(
            item,
            "lighting_camera_realism",
            "lighting",
            "photography_style",
            default=(
                "Use natural photorealistic lighting, soft realistic shadows, lifelike skin texture, "
                "realistic fabric texture, and professional portrait photography quality. The image "
                "should look like a real candid studio, editorial, or lifestyle photograph, not a "
                "painting, not anime, not CGI, and not an overly polished fashion render."
            ),
        )
    )
    anti_drift = normalize_text(
        get_field(
            item,
            "anti_drift_constraints",
            "consistency_requirements",
            default=(
                "Keep the same character identity, same facial structure, same hair color and hairstyle, "
                "same body proportions, and same overall appearance from the reference images. The outfit "
                "should match the outfit description closely. Create a tasteful variation in pose, angle, "
                "expression, and setting while preserving the character."
            ),
        )
    )
    avoid = normalize_text(
        get_field(
            item,
            "avoid",
            "negative_constraints",
            default=(
                "Do not generate a different person. Do not change her face, age, hairstyle, body shape, "
                "or ethnicity. Do not create a collage, contact sheet, illustration, anime image, CGI render, "
                "overly retouched beauty image, watermark, text, logo, or extra person."
            ),
        )
    )

    return "\n\n".join(
        [
            order_block,
            task,
            "The attached reference images define the same adult female character. Use the reference images as the "
            "source of truth for her facial identity, body proportions, skin tone, hair color, hairstyle, and overall "
            "appearance. Preserve her identity with high consistency across all generated images. Do not redesign her face, "
            "do not change her age, do not change her body shape, and do not change her overall visual impression.",
            f"Reference priority:\n{reference_priority}",
            f"Character identity:\n{identity}",
            f"Outfit:\n{outfit}",
            f"Shot and composition:\n{shot_type}.\nCamera view: {camera_view}.\nPose: {pose}.\nExpression: {expression}.",
            f"Scene and environment:\n{environment}",
            f"Lighting and photography style:\n{lighting}",
            f"Consistency requirements:\n{anti_drift}",
            f"Avoid:\n{avoid}",
        ]
    )
