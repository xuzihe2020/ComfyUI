# lora_data_capture

Caption a folder of **face-identity LoRA training images** with **Grok** (xAI
vision model). For every image it writes:

1. `<image_base>.json` in the captures directory — the bilingual caption
   record returned by Grok (**source of truth**, keeps the raw `{TRIGGER}`
   placeholder).
2. `<image_base>.txt` **next to the training image** — the English training
   caption with your `--trigger` token substituted, in the sibling-file format
   ai-toolkit/kohya expect (`my_image.jpeg` + `my_image.txt`).

Standard library only; uses the repo's shared `lib.llm_client.GrokClient`.

## Captioning rules

The prompts live in [`prompts/`](prompts/) — edit them there, not in code:

- `grok_system_caption.txt` — the face-identity captioning contract:
  describe everything changeable (framing, camera angle both axes, head pose,
  micro-expressions, gaze, hands, transient states like sweat/tears/stray
  hairs, detailed hairstyle, clothing, jewellery, lighting, background);
  never describe permanent facial anatomy; refer to her only as `{TRIGGER}`.
- `grok_user_caption.txt` — the per-image user message.

## Structured JSON

Grok must return this shape (enforced via strict `json_schema` + local
validation):

```json
{
  "caption_en": "A close-up portrait of {TRIGGER}, ...",
  "caption_zh": "一张{TRIGGER}的特写肖像，...",
  "multiple_people": false
}
```

Validation rules:

| Field | Rule |
|---|---|
| `caption_en` | Non-empty English (no CJK), contains `{TRIGGER}`, ≥15 words. |
| `caption_zh` | Non-empty Chinese, contains `{TRIGGER}`, same meaning as EN. |
| `multiple_people` | Boolean; `true` when anyone besides her is visible. |

On validation failure the error reason is sent back to Grok (up to
`--retries` times, same behavior as `image_description_v2`). Flagged
`multiple_people` images are listed in the run summary — cull or re-crop them
before training.

## Data flow: JSON is the source of truth

The sibling `.txt` is **derived** and regenerated from the JSON on every run:

- To fix a caption: edit `caption_en` in the JSON, re-run → `.txt` updates.
  Never hand-edit the `.txt`; it will be overwritten.
- To change the trigger token: re-run with a new `--trigger` → all `.txt`
  files update with **zero API calls** (existing JSON is reused).
- To re-caption an image from Grok: `--overwrite`.
- ZH captions exist for human QA review only; they never enter the dataset
  folder as training files.

## Console output & cost estimates

Always on (no flag needed): the console prints one concise line per Grok
request as responses arrive — `request 1: 4021 in / 195 out, ~$0.0055` — and
after the run a total: `Grok usage: 83 request(s), 334021 input tokens, 16800
output tokens, est cost ~$0.4600`.

## Debug logs (`--debug 1`)

With `--debug 1`, the run additionally writes a JSONL log to
`<input_dir>/logs/run_<timestamp>.jsonl` (override the directory with
`--logs-dir`). Each Grok attempt is one line containing:

- `request` — the raw body sent to Grok (model, temperature, messages, response_format),
  with image data URIs replaced by a `<data-uri omitted, N chars>` stub;
- `response` — the raw API response, verbatim (caption content, usage, ids);
- `usage` — input/output token counts from the API `usage` field;
- `est_cost_usd` — estimated cost of that request;
- `validation_error` / `transport_error` when the attempt failed.

The file starts with a `run_start` line (model + prices used) and ends with a
`summary` line (total requests, tokens, cost). The console token/cost output
described above is independent of this file and stays on either way.

Prices live in [`pricing.py`](pricing.py), taken from
[docs.x.ai/docs/models](https://docs.x.ai/docs/models) (checked 2026-07-10):
grok-4.3 at **$1.25/M input, $2.50/M output**; grok-4.5 at $2.00/$6.00. When
xAI changes pricing (or you use an unlisted model), override with
`--price-input` / `--price-output` (USD per million tokens) or update the
table. Estimates bill all prompt tokens at the full input rate — a slight
upper bound when prompt caching applies.

## Setup

```bash
export XAI_API_KEY=sk-...        # or pass --api-key, or put it in the repo .env
```

## Usage

```bash
# Caption every image in the dataset folder; JSON goes to <input>/captures/,
# sibling .txt files go next to the images with the trigger substituted.
python scripts/lora_data_capture/caption_images.py ~/datasets/anna_v1 --trigger anna_zx7

# Custom captures directory
python scripts/lora_data_capture/caption_images.py ~/datasets/anna_v1 --trigger anna_zx7 -o ~/out/captures

# Preview prompts + planned outputs without spending API calls
python scripts/lora_data_capture/caption_images.py ~/datasets/anna_v1 --dry-run

# Change the trigger later — no API calls, .txt files rewritten from JSON
python scripts/lora_data_capture/caption_images.py ~/datasets/anna_v1 --trigger anna_q9k

# Re-caption everything from scratch
python scripts/lora_data_capture/caption_images.py ~/datasets/anna_v1 --trigger anna_zx7 --overwrite
```

If the input folder is `/data/anna_v1/`, the outputs for `my_image.jpeg` are:

```text
/data/anna_v1/captures/my_image.json   # bilingual record, {TRIGGER} kept
/data/anna_v1/my_image.txt             # training caption, trigger substituted
```

## Useful flags

| Flag | Purpose |
|---|---|
| `--trigger TOKEN` | Trigger token written into `.txt` captions. Omitted → `.txt` keeps the literal `{TRIGGER}` placeholder (reminder printed). |
| `-o, --output-dir` | Captures dir for JSON (default `<input_dir>/captures`). |
| `-r, --recursive` | Recurse into subdirectories (JSON names kept unique via `__`; `.txt` stays next to each image). |
| `--overwrite` | Re-caption images that already have JSON. |
| `--limit N` | Process at most N images. |
| `--model` | Vision-capable Grok model id. |
| `--retries {0,1,2,3}` | Validation retries after invalid Grok captions. |
| `--request-retries N` | HTTP retries for transient request errors. |
| `--sleep S` | Sleep S seconds between Grok calls. |
| `--debug 0\|1` | When `1`, write raw request/response JSONL logs. Console token/cost prints are always on. |
| `--logs-dir DIR` | Debug log directory (default `<input_dir>/logs`). |
| `--price-input N` / `--price-output N` | Override USD-per-million-token prices for cost estimates. |
| `--dry-run` | Resolve prompts/inputs, print plan, make no API calls. |

## Module layout

| File | Role |
|---|---|
| `caption_images.py` | CLI entry: argument parsing, batch loop, summary. |
| `captioner.py` | Grok contract: JSON schema, messages, validation, retry loop, `.txt` rendering. |
| `dataset_io.py` | Filesystem: image discovery, data-URI encoding, prompt loading, JSON/`.txt` writing. |
| `run_logging.py` | Per-run JSONL request log + token/cost accounting. |
| `pricing.py` | Grok price table (USD/Mtok) + cost estimation. |
| `prompts/` | System/user prompts, separated from code for easy iteration. |
