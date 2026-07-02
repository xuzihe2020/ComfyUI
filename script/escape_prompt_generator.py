r"""Generate Flux prompt text files from storyboard JSON files.

Run from the ComfyUI repo root:

    python script/escape_prompt_generator.py C:\path\to\example_dir

Expected folder layout:

    example_dir/
      json/
        shot_001.json
        shot_002.json
      prompt/
        shot_001.txt  # generated
        shot_002.txt  # generated

The script reads every `.json` file in `BASE_DIR/json` and writes a matching
`.txt` prompt file into `BASE_DIR/prompt`.
"""

import argparse
import json
from pathlib import Path


PERSPECTIVE_TEXT = {
    "first_person_male_protagonist_pov": "first person POV looking down",
    "third_person_side": "third person side view, no male face visible",
    "third_person_over_shoulder": "over the shoulder view from behind the man",
    "third_person_back": "from behind the man, only his back visible",
}

SHOT_SIZE_TEXT = {
    "full_body": "full body shot",
    "medium_full": "medium full shot (knees up)",
    "medium": "medium shot (waist up)",
    "medium_close_up": "medium close-up (chest up)",
    "close_up": "close-up",
    "extreme_close_up": "extreme close-up",
}

ANGLE_TEXT = {
    "eye_level": "eye level",
    "low_angle": "low angle",
    "high_angle": "high angle",
    "dutch_angle": "dutch angle",
}

FOCUS_TYPE_TEXT = {
    "character": "",
    "body_part_closeup": "extreme close-up of body part, no face visible",
}

DEFAULT_PERSPECTIVE = "first_person_male_protagonist_pov"
DEFAULT_SHOT_SIZE = "medium"
DEFAULT_ANGLE = "eye_level"
DEFAULT_FOCUS_TYPE = "character"

QUALITY_PROMPT = (
    "photorealistic, ultra-detailed skin texture, cinematic lighting, sharp focus, "
    "8k, masterpiece, best quality --ar 16:9"
)


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _enum_text(mapping: dict[str, str], enum_value: object, default_enum_value: str) -> str:
    """Resolve a JSON enum value into the plain prompt text for that enum."""
    return mapping.get(enum_value, mapping[default_enum_value])


def _join_lines(parts: list[object]) -> str:
    return "\n".join(text for part in parts if (text := _text(part)))


def build_scene_prompt_block(scene: dict) -> str:
    """Build the prompt block for the unchanged JSON `scene` object."""
    return _join_lines(
        [
            scene.get("location", ""),
            scene.get("time", ""),
            scene.get("lighting", ""),
            scene.get("environment", ""),
        ]
    )


def build_cinematography_prompt_block(cinematography: dict) -> str:
    """Build the prompt block for the unchanged JSON `cinematography` object."""
    perspective = cinematography.get("perspective", DEFAULT_PERSPECTIVE)
    shot_size = cinematography.get("shot_size", DEFAULT_SHOT_SIZE)
    angle = cinematography.get("angle", DEFAULT_ANGLE)
    focus_type = cinematography.get("focus_type", DEFAULT_FOCUS_TYPE)

    return _join_lines(
        [
            _enum_text(PERSPECTIVE_TEXT, perspective, DEFAULT_PERSPECTIVE),
            _enum_text(SHOT_SIZE_TEXT, shot_size, DEFAULT_SHOT_SIZE),
            _enum_text(FOCUS_TYPE_TEXT, focus_type, DEFAULT_FOCUS_TYPE),
            cinematography.get("body_part", ""),
            _enum_text(ANGLE_TEXT, angle, DEFAULT_ANGLE),
            cinematography.get("composition_notes", ""),
        ]
    )


def build_heroine_prompt_block(heroine: dict, cinematography: dict) -> str:
    """Build the prompt block for the unchanged JSON `heroine` object."""
    focus_type = cinematography.get("focus_type", DEFAULT_FOCUS_TYPE)
    if focus_type == "body_part_closeup":
        return _join_lines(
            [
                heroine.get("clothing_state", ""),
                heroine.get("body_action", ""),
                heroine.get("relationship_to_camera", ""),
            ]
        )

    return _join_lines(
        [
            heroine.get("name", ""),
            heroine.get("proportion_in_frame", ""),
            heroine.get("body", ""),
            heroine.get("face", ""),
            heroine.get("hairstyle", ""),
            heroine.get("clothing_state", ""),
            heroine.get("expression", ""),
            heroine.get("body_action", ""),
            heroine.get("relationship_to_camera", ""),
        ]
    )


def construct_flux_prompt(json_data: dict) -> str:
    """Construct a Flux2 prompt from the existing storyboard JSON structure."""
    cinematography = json_data.get("cinematography", {})
    scene = json_data.get("scene", {})
    heroine = json_data.get("heroine", {})

    return _join_lines(
        [
            build_scene_prompt_block(scene),
            build_cinematography_prompt_block(cinematography),
            build_heroine_prompt_block(heroine, cinematography),
            QUALITY_PROMPT,
        ]
    )


def generate_prompt_files(base_dir: Path) -> int:
    json_dir = base_dir / "json"
    prompt_dir = base_dir / "prompt"

    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON input directory does not exist: {json_dir}")

    prompt_dir.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json files found in: {json_dir}")

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as file:
            json_data = json.load(file)
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected a JSON object in: {json_path}")

        prompt = construct_flux_prompt(json_data)
        prompt_path = prompt_dir / json_path.with_suffix(".txt").name
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        print(f"{json_path.name} -> {prompt_path}")

    return len(json_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Flux prompt .txt files from JSON objects in BASE_DIR/json, "
            "writing matching filenames to BASE_DIR/prompt."
        )
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing a json/ folder; prompt/ will be created beside it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_prompt_files(args.base_dir)
    print(f"Generated {count} prompt file(s).")


if __name__ == "__main__":
    main()
