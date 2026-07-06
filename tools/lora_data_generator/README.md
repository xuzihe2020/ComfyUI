# LoRA Data Generator

Generate synthetic LoRA training images from structured job JSON with either of
two backends behind one CLI:

| Mode | Backend | Transport |
|---|---|---|
| `flux2-max` | FLUX.2 [max] via the official BFL API (`POST api.bfl.ai/v1/flux-2-max`) | async submit + poll per image |
| `gpt-image-2` | OpenAI `gpt-image-2` via `POST /v1/images/edits` | direct sync requests, or the OpenAI Batch API at 50% token rates |

Neither pipeline requires ComfyUI: both call their vendor's REST API directly,
with usage visible in the vendor consoles (dashboard.bfl.ai / platform.openai.com).
The old ComfyUI canvas workflow under `user/default/workflows/prod/lora_references/`
still works interactively in the GUI, but this tool no longer uses it.

Prompt construction (`tool_lib/prompting.py`) and reference-image resolution and
indexing (`tool_lib/references.py`) are shared by both pipelines. A run with the
same `--input-json` sends **byte-identical prompt text** and the **same
reference images in the same order** to both models, so output comparisons are
apples-to-apples. The GPT pipeline deliberately uses the Images API rather than
the Responses API image tool, because the latter lets the mainline model
rewrite the prompt before the image model sees it.

## Layout

```
main.py                       CLI entry point; --mode flux2-max | gpt-image-2
tool_lib/paths.py             repo root + JSON helpers
tool_lib/jobs.py              job JSON normalization, field access, output naming
tool_lib/references.py        reference path resolution, ordering, limits
tool_lib/prompting.py         the shared prompt builder (single source of truth)
components/flux2_max_runner.py  FLUX.2 Max pipeline (direct BFL API)
components/gpt_image2_runner.py GPT Image 2 pipeline (sync + batch + fetch)
```

API clients live in the repo-root `lib/llm_client/` package (BFL and OpenAI
clients subclass `lib/llm_client/base.py`), and .env/API-key loading in
`lib/envfile.py` — both shared by all repo scripts and tools.

No third-party dependencies; runs with the repo venv python.

## Credentials

`main.py` loads API keys from the repo `.env` (override the file with
`--env-file`); real environment variables take precedence over `.env` values.

- `flux2-max`: `FLUX_API_KEY` (alias `BFL_API_KEY`) — a Black Forest Labs key
  from https://dashboard.bfl.ai (Projects -> API -> Keys; prepaid credits,
  1 credit = $0.01)
- `gpt-image-2`: `OPENAI_API_KEY` — from https://platform.openai.com

## Usage

Dry-run first (builds prompts and debug logs, calls no API):

```
python tools/lora_data_generator/main.py --mode flux2-max --input-json jobs.json --dry-run
python tools/lora_data_generator/main.py --mode gpt-image-2 --input-json jobs.json --dry-run
```

FLUX.2 Max via the BFL API (each run prints the actual billed cost and
input/output megapixels reported by BFL):

```
python tools/lora_data_generator/main.py --mode flux2-max --input-json jobs.json --limit 5 --repeat 2
```

GPT Image 2, synchronous:

```
python tools/lora_data_generator/main.py --mode gpt-image-2 --input-json jobs.json --limit 5 --repeat 2
```

GPT Image 2 through the Batch API (50% rates, completes within 24h):

```
# submit and poll until done (or until --timeout, then resume later):
python tools/lora_data_generator/main.py --mode gpt-image-2 --input-json jobs.json --transport batch

# submit and exit immediately:
python tools/lora_data_generator/main.py --mode gpt-image-2 --input-json jobs.json --transport batch --no-wait

# fetch results later (batch id is printed at submit time and saved to submitted.json):
python tools/lora_data_generator/main.py --mode gpt-image-2 --fetch-batch batch_abc123
```

Common flags: `--limit N`, `--repeat N`, `--width/--height` (defaults defined in
`main.py`: 1024x1536), `--env-file`, `--output-dir`, `--output-format`,
`--log-dir`, `--no-log`, `--dry-run`, `--timeout`,
`--no-wait` (gpt batch only).
Flux-only: `--seed`, `--safety-tolerance` (BFL moderation, 0 strictest to 5),
`--bfl-base-url`.
GPT-only: `--transport`, `--size` (overrides `--width/--height`; includes `auto`),
`--quality` (default `high`), `--moderation`, `--poll-interval`,
`--fetch-batch`, `--base-url`.

This tool replaces the former `scripts/workflows/run_flux2_max_lora_references.py`;
use `main.py --mode flux2-max` instead.

## Input JSON format

The file passed to `--input-json` may be a single job object, a list of jobs,
or an object with an `items`/`jobs`/`prompts` list.

```json
{
  "output_stem": "ol_office_halfbody_front_0001",
  "dressing_reference_images": ["refs/outfit_front.png", "refs/outfit_detail.png"],
  "character_reference_images": [
    "refs/face_front.png",
    "refs/face_3q_left.png",
    "refs/face_3q_right.png",
    "refs/body_front.png",
    "refs/body_side.png"
  ],
  "outfit_block": "Office lady outfit: fitted navy blazer, white blouse, pencil skirt.",
  "shot_type": "Half-body portrait",
  "camera_view": "front 3/4 view",
  "pose": "standing naturally with relaxed shoulders",
  "expression": "soft confident smile",
  "environment_block": "Modern office interior, softly blurred background.",
  "lighting_camera_realism": "Soft window light from camera left, 85mm portrait lens look."
}
```

References: up to 2 `dressing_reference_images` (aliases `dressing_references`,
`dressing_refs`) and up to 5 `character_reference_images` (aliases
`character_references`, `character_refs`). Those are maximums — any smaller
count (including zero) works in both pipelines: flux attaches only the given
images as `input_image(_N)` fields (BFL allows up to 8), and gpt-image-2
attaches only the given images (a job with no references falls back from
`/images/edits` to `/images/generations`, and batch runs are grouped into one
batch per endpoint). Paths may be absolute, relative to the input JSON, or
relative to the repo root. Extensions: `.png .jpg .jpeg .webp` (`.bmp` is
accepted at resolution time but rejected by both APIs).

References are indexed dressing-first then character, and the shared prompt
opens with an "Attached reference image order" block that tells the model what
"Reference image N" means. Both pipelines attach the actual images in exactly
that order.

Prompt fields (all optional; built-in master-template defaults fill gaps, and
fields may be nested under `"chunks"`): `task_output_goal`/`task`/`output_goal`,
`reference_priority`, `identity_lock`/`character_identity`/`identity`,
`outfit_block`/`outfit`, `shot_type`/`shot`/`framing`, `camera_view`/`angle`,
`pose`/`action`, `expression`, `environment_block`/`environment`/`background`,
`lighting_camera_realism`/`lighting`/`photography_style`,
`anti_drift_constraints`/`consistency_requirements`,
`avoid`/`negative_constraints`. Providing `prompt`/`positive_prompt` bypasses
chunk assembly (the reference-order block is still prepended).

Per-job overrides: `output_stem`/`name`/`id`, `width`, `height`, `size`
(gpt only), and `seed` (flux only).

Output size resolution: per-job `size` > per-job `width`/`height` > `--size`
(gpt only) > `--width`/`--height` (defaults 1024x1536 from `main.py`).
gpt-image-2 accepts only `1024x1024`, `1024x1536`, `1536x1024`, or `auto`;
the BFL API takes free-form width/height (min 64, up to 4MP total).

## Outputs and logs

Final images from both pipelines are written to `--output-dir` (default
`output/lora_data_generator/`, repo-relative unless absolute), named:

```
{input_json_stem}_{flux2|gpt}_{unix_seconds}.{png|jpeg|webp}
```

A `_NN` counter suffix de-duplicates same-second saves.

- flux2-max: each BFL task is polled until Ready, then the signed result URL
  (valid ~10 minutes) is downloaded into `--output-dir`. The debug log records
  the request id plus the actual billed `cost` / `input_mp` / `output_mp` from
  the submit response.
- gpt-image-2: images are written directly to `--output-dir` (sync, batch wait,
  and `--fetch-batch`).

Per-request debug logs mirror each other under `logs/flux2_max_lora_references/`
and `logs/gpt_image2_lora_references/`: final prompt, ordered references with
role labels, size/seed/quality, and (GPT) the raw `usage` block plus an
estimated USD cost. API keys are never written to logs.

Batch state lives in `logs/gpt_image2_lora_references/batches/<ts>_<input>/`:
`requests.jsonl` (uploaded bodies), `manifest.json` (custom_id -> job mapping),
`submitted.json` (batch id), `results.json` (post-fetch summary).

## Caching and cost notes (gpt-image-2)

- OpenAI prices a cached-input rate for `gpt-image-2` ($2/1M vs $8/1M image
  input; half that on batch). The pipeline maximizes cacheability by sending
  byte-identical reference bytes in a stable order, with the stable prompt
  prefix (reference-order block) first.
- Whether a given request actually hit the cache is reported per request:
  check `usage.input_tokens_details.cached_tokens` in the debug logs / batch
  results, and the printed per-image estimated cost.
- Cost estimates use the published gpt-image-2 rates (text in $5, image in $8,
  cached in $2, image out $30 per 1M tokens; Batch = 50%). They are estimates;
  the raw usage block is always logged alongside.
- Batch JSONL embeds references as base64 data URLs (multipart uploads are not
  available inside batch bodies), so request files are large: roughly 1.4 MB
  per job with 7 full-res references. The OpenAI batch input limit is 200 MB;
  split big runs with `--limit` if needed.
