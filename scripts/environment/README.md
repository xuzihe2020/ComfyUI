# Script Utilities

## Gemini Environment Briefs

Batch-extract structured visual novel environment briefs from reference images:

```bash
export GEMINI_API_KEY="your-key"

python3 script/gemini_environment_briefs.py /path/to/reference_images \
  --output-dir output/environment_briefs \
  --recursive \
  --model gemini-2.5-flash
```

By default the script extracts a richer JSON brief but renders a compact prompt,
so the image model is not over-constrained. The compact prompt includes
`environment_type`, `scene_location`, `has_windows`, `style_description`,
`required_items`, and `avoid`; it keeps `character_safe_zone` and
`identity_anchors` in JSON metadata and expresses them softly in prose. Use
`--prompt-detail full` to include more fields directly in the rendered prompt.

Gemini also classifies whether the reference is indoor or outdoor, and whether
an indoor scene has exterior windows. The script uses that metadata to render
three extra lighting/time variants for each image:

- Outdoor: morning, afternoon, evening with practical lights on
- Indoor with windows: daytime curtains/blinds open, night curtains/blinds open
  with room lights on, curtains/blinds closed with only indoor light
- Indoor without windows: bright indoor light, warm indoor light, dim indoor
  light

The target rendering style is fixed by the script, not inferred from Gemini.
Default:

```text
3D Unity CG rendering style, like a game environment rendered in Unity, clean
realtime 3D scene, coherent geometry, polished game-environment materials, soft
baked lighting, not photorealistic
```

Override it only if the whole project style changes:

```bash
python3 script/gemini_environment_briefs.py /path/to/reference_images \
  --rendering-style "3D Unity CG rendering style, like a game environment rendered in Unity"
```

The generated prompt tells the image model to ignore/remove watermarks,
signatures, stock-photo marks, borders, captions, and UI overlays from the
reference image.

Outputs:

- `briefs/**/*.environment.json`: source image, extracted JSON brief, rendered base prompt
- `prompts/**/*.base_prompt.txt`: prompt ready to use with the reference image
- `prompts/**/*.<variant>.base_prompt.txt`: prompt variants ready to test time
  and lighting consistency
- `manifest.jsonl`: batch index for downstream automation

Dry run:

```bash
python3 script/gemini_environment_briefs.py /path/to/reference_images --dry-run
```

The script calls Gemini directly through REST, not through ComfyUI nodes. It is
intended to produce text/JSON assets that a ComfyUI workflow can consume.

## OpenAI Environment Image Generation

Generate images from the environment brief folder using the source reference
image plus each prompt:

```bash
export OPENAI_API_KEY="your-key"

python3 script/openai_generate_environment_images.py \
  /Users/tonyxu/Documents/backgrounds/environment_briefs \
  /Users/tonyxu/Documents/backgrounds/openai_generated \
  --images-per-prompt 1
```

This uses `gpt-image-2`, reads `manifest.jsonl`, and generates every prompt
attached to every source image. The script treats the neutral prompt and all
lighting/time prompts as equal jobs.

```text
openai_generated/
  bedroom_01/
    neutral/
      bedroom_01.neutral.a1b2c3d4-f9e8a7.png
      bedroom_01.neutral.a1b2c3d4-f9e8a7.png.json
    daytime_curtains_open/
      bedroom_01.daytime_curtains_open.a1b2c3d4-111aaa.png
      bedroom_01.daytime_curtains_open.a1b2c3d4-111aaa.png.json
  generation_manifest.jsonl
```

The suffix is random per script run and per output image. In the example above,
`a1b2c3d4` is the run ID and `f9e8a7` is the image ID, so repeated runs do not
overwrite earlier candidates.

By default, reruns skip a prompt if the output folder already contains any
generated image for that source image and prompt. This lets interrupted batches
resume without repeating finished prompt jobs. Earlier outputs written under
`base/` are treated as existing `neutral` prompt outputs.

Useful options:

```bash
# Generate three outputs for each prompt.
python3 script/openai_generate_environment_images.py BRIEF_DIR OUT_DIR \
  --images-per-prompt 3

# Stop after 10 total output images. Omit --max-images to process everything.
python3 script/openai_generate_environment_images.py BRIEF_DIR OUT_DIR \
  --images-per-prompt 3 \
  --max-images 10

# Dry run without calling OpenAI.
python3 script/openai_generate_environment_images.py BRIEF_DIR OUT_DIR \
  --images-per-prompt 2 \
  --max-images 8 \
  --dry-run

# Optional: use your own run ID in output filenames.
python3 script/openai_generate_environment_images.py BRIEF_DIR OUT_DIR \
  --run-id test01

# Force regeneration even when a prompt already has output images.
python3 script/openai_generate_environment_images.py BRIEF_DIR OUT_DIR \
  --rerun-existing-prompts
```

Important defaults:

- `--model gpt-image-2`
- `--size 1536x864`
- `--quality high`
- `--output-format png`
- `--background opaque`

The script calls OpenAI's image edit endpoint because each prompt is paired with
a reference image. It writes one `.json` metadata file next to each generated
image and appends the same records to `generation_manifest.jsonl`.
