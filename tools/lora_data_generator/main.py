#!/usr/bin/env python3
"""Generate synthetic LoRA training images from structured job JSON.

One CLI, two backends selected with --mode:

- flux2-max:    calls the official BFL API (api.bfl.ai) directly with a key
                from https://dashboard.bfl.ai; async submit + poll per image.
- gpt-image-2:  calls the OpenAI Images API directly, either synchronously or
                through the 50%-priced Batch API (--transport batch).

Both modes share the same input JSON format, prompt construction, and
reference image ordering (lib/), so a run with the same --input-json is an
apples-to-apples comparison between the two models.

Examples
--------
Dry-run either pipeline (build prompts and logs, no API calls):

    python tools/lora_data_generator/main.py --mode flux2-max \
        --input-json path/to/jobs.json --dry-run
    python tools/lora_data_generator/main.py --mode gpt-image-2 \
        --input-json path/to/jobs.json --dry-run

Flux.2 Max via the BFL API (needs FLUX_API_KEY from dashboard.bfl.ai in .env):

    python tools/lora_data_generator/main.py --mode flux2-max \
        --input-json path/to/jobs.json --limit 5 --repeat 2

GPT Image 2, synchronous (needs OPENAI_API_KEY in .env):

    python tools/lora_data_generator/main.py --mode gpt-image-2 \
        --input-json path/to/jobs.json --limit 5 --repeat 2

GPT Image 2 through the Batch API, submit now and fetch later:

    python tools/lora_data_generator/main.py --mode gpt-image-2 \
        --input-json path/to/jobs.json --transport batch --no-wait
    python tools/lora_data_generator/main.py --mode gpt-image-2 \
        --fetch-batch batch_abc123

After generation, every saved image is face-scored (ArcFace cosine similarity)
against the item's primary character reference — the first character reference
image — printed as `<image name>: <score>` and written to the debug log.

Face-verify two images without generating anything (utility mode):

    python tools/lora_data_generator/main.py --mode verify \
        --images ref_closeup.png generated_fullbody.png

See tools/lora_data_generator/README.md for the input JSON format and details.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `components`/`tool_lib` (tool-local) and the repo-root `lib` package
# (shared API clients, .env loading) importable regardless of the caller's CWD.
_TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOL_DIR.parents[1]))
sys.path.insert(0, str(_TOOL_DIR))

from components import face_verify, flux2_klein_local_runner, flux2_max_runner, gpt_image2_runner  # noqa: E402
from lib.envfile import DEFAULT_ENV_FILE, env_api_key, load_env_file  # noqa: E402
from lib.llm_client import BFLClient, OpenAIClient  # noqa: E402

MODES = ("flux2-max", "gpt-image-2", "flux2-klein-local", "verify")

# Default output size for both pipelines: 2:3 portrait.
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1536

# Final images from both pipelines land here (repo-relative unless absolute),
# named {input_json_stem}_{flux2|gpt}_{unix_seconds}.<ext>.
DEFAULT_OUTPUT_DIR = Path("output/lora_data_generator")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora_data_generator",
        description=(
            "Generate synthetic LoRA training images via FLUX.2 Max (BFL API), "
            "GPT Image 2 (OpenAI API), or a local FLUX.2 Klein ComfyUI workflow."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        required=True,
        help="Generation backend, or 'verify' to face-score two images without generating.",
    )
    parser.add_argument(
        "--face-model-root",
        help="Directory for the InsightFace model pack (~330 MB); models land under "
        "<root>/models/buffalo_l. Default: ~/.insightface (or FACE_MODEL_ROOT in .env).",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Job JSON file, or a directory of .json job files. Each file may be a single object, list, or items/jobs/prompts.",
    )
    parser.add_argument("--limit", type=int, help="Process at most N jobs. Default: unlimited.")
    parser.add_argument(
        "--repeat",
        "--repeat-count",
        dest="repeat",
        type=int,
        default=1,
        help="Queue each selected prompt N times. Default: 1.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Output width. Default: {DEFAULT_WIDTH} for 2:3 portrait.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Output height. Default: {DEFAULT_HEIGHT} for 2:3 portrait.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Env file to load API keys from (FLUX_API_KEY / OPENAI_API_KEY). Default: repo .env. "
        "Real environment variables take precedence.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where final images are written, named "
        "{input_json_stem}_{flux2|gpt|klein}_{unix_seconds}.<ext>. "
        "Repo-relative unless absolute. Default: %(default)s.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Write per-request debug logs here. "
        f"Defaults: {flux2_max_runner.DEFAULT_LOG_DIR} / {gpt_image2_runner.DEFAULT_LOG_DIR} / "
        f"{flux2_klein_local_runner.DEFAULT_LOG_DIR}.",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable per-request debug logs.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and logs without calling any API.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to wait for completion (per BFL task, per sync request, or per batch). Default: 3600.",
    )
    parser.add_argument("--no-wait", action="store_true", help="gpt-image-2 batch only: submit without waiting.")
    parser.add_argument(
        "--output-format",
        choices=("png", "jpeg", "webp"),
        default="png",
        help="Image format for both backends. Default: png.",
    )

    flux = parser.add_argument_group("flux2-max options (direct BFL API)")
    flux.add_argument("--seed", type=int, help="Fixed seed for the whole run. Default: random per request.")
    flux.add_argument(
        "--safety-tolerance",
        type=int,
        choices=range(0, 6),
        help="BFL moderation strictness, 0 (strictest) to 5. Omitted from the request unless set (API default: 2).",
    )
    flux.add_argument("--bfl-base-url", default=BFLClient.DEFAULT_BASE_URL, help="BFL API base URL.")

    gpt = parser.add_argument_group("gpt-image-2 options")
    gpt.add_argument("--transport", choices=("sync", "batch"), default="sync", help="Direct requests or Batch API. Default: sync.")
    gpt.add_argument(
        "--size",
        choices=sorted(gpt_image2_runner.SUPPORTED_SIZES) + ["auto"],
        help="Output size for gpt-image-2; overrides --width/--height. Default: derived from --width/--height.",
    )
    gpt.add_argument("--quality", choices=("low", "medium", "high", "auto"), default="high", help="Default: high.")
    gpt.add_argument(
        "--moderation",
        choices=("auto", "low"),
        help="Content moderation strictness. Omitted from the request unless set.",
    )
    gpt.add_argument("--poll-interval", type=int, default=60, help="Batch status poll interval in seconds. Default: 60.")
    gpt.add_argument("--fetch-batch", metavar="BATCH_ID", help="Fetch results of a previously submitted batch and exit.")
    gpt.add_argument("--base-url", default=OpenAIClient.DEFAULT_BASE_URL, help="OpenAI API base URL.")

    local = parser.add_argument_group("flux2-klein-local options (running ComfyUI server)")
    local.add_argument(
        "--server",
        default=flux2_klein_local_runner.DEFAULT_SERVER,
        help="ComfyUI server URL. Default: %(default)s.",
    )
    local.add_argument(
        "--workflow",
        type=Path,
        default=flux2_klein_local_runner.DEFAULT_WORKFLOW,
        help="ComfyUI UI workflow JSON to run. Default: %(default)s.",
    )
    local.add_argument(
        "--steps",
        type=int,
        default=flux2_klein_local_runner.DEFAULT_STEPS,
        help="Local Flux2Scheduler steps. Default: %(default)s.",
    )
    local.add_argument(
        "--guidance",
        type=float,
        default=flux2_klein_local_runner.DEFAULT_GUIDANCE,
        help="Local FluxGuidance value. Default: %(default)s.",
    )
    local.add_argument(
        "--sampler-name",
        default=flux2_klein_local_runner.DEFAULT_SAMPLER_NAME,
        help="Local KSamplerSelect sampler_name. Default: %(default)s.",
    )
    local.add_argument(
        "--reference-method",
        default=flux2_klein_local_runner.DEFAULT_REFERENCE_METHOD,
        choices=("offset", "index", "uxo/uno", "index_timestep_zero"),
        help="Local Flux reference latent method. Default: %(default)s.",
    )
    local.add_argument("--klein-unet-name", default=flux2_klein_local_runner.DEFAULT_UNET_NAME)
    local.add_argument("--klein-weight-dtype", default=flux2_klein_local_runner.DEFAULT_WEIGHT_DTYPE)
    local.add_argument("--klein-clip-name", default=flux2_klein_local_runner.DEFAULT_CLIP_NAME)
    local.add_argument("--klein-clip-type", default=flux2_klein_local_runner.DEFAULT_CLIP_TYPE)
    local.add_argument("--klein-clip-device", default=flux2_klein_local_runner.DEFAULT_DEVICE)
    local.add_argument("--klein-vae-name", default=flux2_klein_local_runner.DEFAULT_VAE_NAME)

    verify = parser.add_argument_group("verify options (face similarity utility)")
    verify.add_argument(
        "--images",
        nargs=2,
        type=Path,
        metavar=("IMAGE_A", "IMAGE_B"),
        help="Two images to face-score against each other (--mode verify).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "verify":
        if not args.images:
            raise SystemExit("--mode verify requires --images IMAGE_A IMAGE_B.")
        load_env_file(args.env_file)  # picks up FACE_MODEL_ROOT if set
        return face_verify.verify_pair(args.images[0], args.images[1], args.face_model_root)

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive when provided.")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive.")
    if args.input_json is None and not (args.mode == "gpt-image-2" and args.fetch_batch):
        raise SystemExit("--input-json is required (except with --mode gpt-image-2 --fetch-batch).")

    load_env_file(args.env_file)

    if args.mode == "flux2-max":
        args.flux_api_key = env_api_key(*BFLClient.ENV_KEYS)
        if not args.flux_api_key and not args.dry_run:
            keys = " or ".join(BFLClient.ENV_KEYS)
            raise SystemExit(
                f"Missing BFL API key (from https://dashboard.bfl.ai). "
                f"Set {keys} in {args.env_file} or the environment."
            )
        if args.no_wait:
            print("note: --no-wait is gpt-image-2 batch only; flux2-max polls each task", file=sys.stderr)
        return flux2_max_runner.run(args)

    if args.mode == "flux2-klein-local":
        if args.no_wait:
            print("note: --no-wait is gpt-image-2 batch only; flux2-klein-local waits for ComfyUI history", file=sys.stderr)
        if args.size:
            print("note: --size is gpt-image-2 only; flux2-klein-local uses --width/--height or per-job width/height", file=sys.stderr)
        return flux2_klein_local_runner.run(args)

    args.openai_api_key = env_api_key(*OpenAIClient.ENV_KEYS)
    needs_key = not args.dry_run or bool(args.fetch_batch)
    if not args.openai_api_key and needs_key:
        raise SystemExit(f"Missing OpenAI API key. Set OPENAI_API_KEY in {args.env_file} or the environment.")
    if args.seed is not None:
        print("note: gpt-image-2 does not support seeds; ignoring --seed", file=sys.stderr)
    if args.no_wait and args.transport == "sync" and not args.fetch_batch:
        print("note: --no-wait has no effect with --transport sync", file=sys.stderr)
    return gpt_image2_runner.run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
