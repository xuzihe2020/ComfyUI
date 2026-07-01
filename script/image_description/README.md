# image_description

Describe a directory of images with **Grok** (xAI vision model), then build a
**FLUX.2** repaint prompt for each one.

For every image it:

1. Sends the image + an instruction prompt to a Grok vision model and gets back a
   **strict structured JSON** description.
2. Writes that JSON to `<base>.json`.
3. Deterministically assembles a FLUX.2 text-to-image prompt from the JSON (with
   optional prefix/suffix blocks) and writes it to `<base>.flux2.txt`.

Standard library only — no `pip install` needed.

## Structured JSON fields

```jsonc
{
  "scene_description":        "…",   // 场景描述: setting, subjects, action, composition
  "environment_and_lighting": "…",   // 环境和灯光: location, props, light sources/mood
  "camera_and_perspective":   "…",   // 镜头视角: shot size, angle, lens, DoF, framing
  "character_relationships":  "…",   // 人物关系: spatial/social relations between people
  "characters": [                     // 具体人物的描述: one entry per visible person
    {
      "label":              "…",     // short handle, e.g. "foreground woman"
      "appearance":         "…",     // 外貌
      "clothing":           "…",     // 衣着
      "body_and_action":    "…",     // 肢体动作
      "relation_to_camera": "…"      // 与镜头的关系/视角
    }
  ]
}
```

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
  --suffix "shot on Hasselblad X2D, 80mm, f/2.8, natural color grade"

# Prefix/suffix from files instead of inline strings
python script/image_description/describe_images.py ./shots \
  --prefix-file style_prefix.txt --suffix-file style_suffix.txt

# Preview prompts + planned outputs without spending any API calls
python script/image_description/describe_images.py ./shots --dry-run
```

### Useful flags

| Flag | Purpose |
|---|---|
| `-o, --output-dir` | Output dir (default `<input_dir>/descriptions`). |
| `-r, --recursive` | Recurse into subdirectories (names kept unique via `__`). |
| `--overwrite` | Re-run images that already have output. |
| `--limit N` | Process at most N images (debugging). |
| `--model` | Vision-capable Grok model id (default `grok-4.3`). |
| `--language` | Language for description values (default `English`). |
| `--prefix / --prefix-file` | Text prepended to every FLUX.2 prompt. |
| `--suffix / --suffix-file` | Text appended to every FLUX.2 prompt. |
| `--sleep S` | Sleep S seconds between images (rate-limit friendly). |
| `--dry-run` | Resolve prompts/inputs, print plan, make no API calls. |

## Outputs

Per image, under the output dir:

- `<base>.json` — the structured description from Grok.
- `<base>.flux2.txt` — the assembled FLUX.2 prompt.
- `<base>.error.txt` — written only if that image failed.

Per run:

- `_effective_grok_system.txt`, `_effective_grok_user.txt` — the exact prompts sent
  to Grok (after `{language}` substitution), for review/debugging.
- `_run_meta.json` — model, settings, prefix/suffix, image count.

## The prompts sent to Grok

They live as editable text blobs in [`prompts/`](prompts/):

- `grok_system.txt` — analyst role + rules (specificity, hex colors, lighting, etc.).
- `grok_user.txt` — the per-field description instructions.

`{language}` is the only placeholder; it is filled from `--language`.

## FLUX.2 prompt construction

Built by **static string assembly** (no model call). Segment order follows FLUX.2
guidance — front-loaded subject, prose (not keyword soup), explicit lighting/camera,
and **no negative prompts**:

```
<prefix>

<scene_description>. <character clauses…>. <character_relationships>.
<environment_and_lighting>. <camera_and_perspective>.

<suffix>
```

Each character clause is `label: appearance, wearing clothing, action, relation_to_camera`.
Edit `build_flux2_prompt()` in `describe_images.py` to change ordering or wording.

References: [FLUX.2 prompting guide](https://docs.bfl.ml/guides/prompting_guide_flux2),
[xAI structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs).
