# image_description_v2

Describe a directory of images with **Grok** (xAI vision model), validate the
returned JSON, and write one `.json` file per image.

For every image it:

1. Sends the image plus instruction prompts to Grok.
2. Asks Grok for the v2 JSON shape.
3. Validates the output locally.
4. If invalid, sends the validation error back to Grok and retries.
5. Writes `<image_name>.json` to `<input_dir>/descriptions` by default.

Standard library only; no `pip install` needed.

The script has two prompt variants:

- Standard prompts are used by default.
- Mosaic prompts are enabled with `--enable-mosaic-version`.

## Structured JSON

Grok must return this shape:

```json
{
  "cinematography": "string",
  "scene": "string",
  "heroine": "string",
  "genre": "string or null",
  "flux_prompt": "string"
}
```

Validation rules:

| Field | Rule |
|---|---|
| `cinematography` | Required non-empty string. |
| `scene` | Required non-empty string. |
| `heroine` | Required non-empty string. |
| `genre` | Required string or `null`. |
| `flux_prompt` | Required non-empty string. |

No extra fields are allowed.

## Retry Behavior

`--retries` controls validation retries after Grok returns invalid JSON. The
maximum is 3.

On each validation retry, the tool logs the reason and sends that reason back to
Grok, for example:

```text
! invalid Grok JSON: scene field is missing, it must present
... validation retry 1/3: sending error reason back to Grok
```

If all validation retries fail for an image, the tool exits.

`--request-retries` is separate and only controls transient HTTP request retries.

## Setup

```bash
export XAI_API_KEY=sk-...        # or pass --api-key
```

PowerShell:

```powershell
$env:XAI_API_KEY = "sk-..."
```

## Usage

```bash
# Basic: describe every image in ./shots, write outputs to ./shots/descriptions
python scripts/image_description_v2/describe_images.py ./shots

# Custom output dir
python scripts/image_description_v2/describe_images.py ./shots -o ./out/descriptions

# Recurse into subdirectories and overwrite existing JSON files
python scripts/image_description_v2/describe_images.py ./shots -r --overwrite

# Use fewer validation retries
python scripts/image_description_v2/describe_images.py ./shots --retries 1

# Use the mosaic-specific prompt variant
python scripts/image_description_v2/describe_images.py ./shots --enable-mosaic-version

# Write the prompts sent to Grok to descriptions/logs.txt
python scripts/image_description_v2/describe_images.py ./shots --debug 1

# Preview prompts + planned outputs without spending API calls
python scripts/image_description_v2/describe_images.py ./shots --dry-run
```

If the input folder is `/this/is/my/images/`, the default output path for
`my_image_name.png` is:

```text
/this/is/my/images/descriptions/my_image_name.json
```

If `-o /this/is/my/output/descriptions/` is supplied, the output path is:

```text
/this/is/my/output/descriptions/my_image_name.json
```

## Useful Flags

| Flag | Purpose |
|---|---|
| `-o, --output-dir` | Output dir (default `<input_dir>/descriptions`). |
| `-r, --recursive` | Recurse into subdirectories (names kept unique via `__`). |
| `--overwrite` | Re-run images that already have output. |
| `--limit N` | Process at most N images. |
| `--model` | Vision-capable Grok model id. |
| `--language` | Language for text values. |
| `--enable-mosaic-version` | Use `grok_system_mosaic.txt` and `grok_user_mosaic.txt` instead of the standard prompts. |
| `--debug 0\|1` | When `1`, write prompt messages sent to Grok to `logs.txt` in the output directory. Image data URIs are omitted from the log. |
| `--retries {0,1,2,3}` | Validation retries after invalid Grok JSON. |
| `--request-retries N` | HTTP retries for transient request errors. |
| `--sleep S` | Sleep S seconds between images. |
| `--dry-run` | Resolve prompts/inputs, print plan, make no API calls. |
