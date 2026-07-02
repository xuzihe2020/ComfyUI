# image_description

Describe a directory of images with **Grok** (xAI vision model), then build a
compact **FLUX.2** prompt for each one.

For every image it:

1. Sends the image plus instruction prompts to Grok and asks for strict JSON.
2. Writes that JSON to `<base>.json`.
3. Builds a FLUX.2 prompt from the JSON and writes `<base>.flux2.txt`.

Standard library only; no `pip install` needed.

## Structured JSON

Grok returns this shape:

```jsonc
{
  "cinematography": {
    "perspective": "first_person_male_protagonist_pov | third_person_side | third_person_over_shoulder | third_person_back",
    "shot_size": "full_body | medium_full | medium | medium_close_up | close_up | extreme_close_up",
    "focus_type": "character | body_part_closeup",
    "body_part": "hands or null",
    "angle": "eye_level | low_angle | high_angle | dutch_angle",
    "composition_notes": "6-18 words"
  },
  "scene": {
    "location": "3-10 words",
    "time": "1-5 words",
    "lighting": "4-12 words",
    "environment": "6-18 words"
  },
  "heroine": {
    "name": "1-4 words",
    "proportion_in_frame": "4-10 words",
    "body": "6-16 words",
    "face": "6-14 words or null",
    "hairstyle": "3-10 words or null",
    "clothing_state": "6-16 words",
    "expression": "3-10 words or null",
    "body_action": "6-16 words",
    "relationship_to_camera": "5-14 words"
  }
}
```

The word limits are guidance for Grok so the final FLUX.2 prompt stays compact.

## Setup

```bash
export XAI_API_KEY=sk-...        # or pass --api-key
```

## Usage

```bash
# Basic: describe every image in ./shots, write outputs to ./shots/descriptions
python script/image_description/describe_images.py ./shots

# Custom output dir, recurse, add FLUX.2 prefix/suffix blocks
python script/image_description/describe_images.py ./shots \
  -o ./out -r \
  --prefix "cinematic film still, photorealistic" \
  --suffix "shot on Hasselblad X2D, natural color grade"

# Preview prompts + planned outputs without spending API calls
python script/image_description/describe_images.py ./shots --dry-run
```

## Useful Flags

| Flag | Purpose |
|---|---|
| `-o, --output-dir` | Output dir (default `<input_dir>/descriptions`). |
| `-r, --recursive` | Recurse into subdirectories (names kept unique via `__`). |
| `--overwrite` | Re-run images that already have output. |
| `--limit N` | Process at most N images. |
| `--model` | Vision-capable Grok model id. |
| `--language` | Language for non-enum description values. |
| `--prefix / --prefix-file` | Text prepended to every FLUX.2 prompt. |
| `--suffix / --suffix-file` | Text appended to every FLUX.2 prompt. |
| `--sleep S` | Sleep S seconds between images. |
| `--dry-run` | Resolve prompts/inputs, print plan, make no API calls. |

## FLUX.2 Prompt Construction

Built by static string assembly; no second model call is involved. Each block uses
comma-separated lines, with a period on the final line:

```text
scene location,
scene time,
scene lighting,
scene environment.

camera perspective,
shot size,
focus mode,
camera angle,
composition notes.

heroine identity,
frame proportion,
body,
face,
hair,
clothing,
expression,
action,
camera relation.

photorealistic, ultra-detailed skin texture, cinematic lighting, sharp focus, 8k, masterpiece, best quality --ar 16:9
```

The prompt text is built in `build_flux2_prompt()` in `describe_images.py`.
