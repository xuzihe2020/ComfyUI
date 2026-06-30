#!/usr/bin/env python3
"""Sample JPEG frames from long videos over optional time clips.

Examples
--------
Sample 1 frame/sec over two clips from every video in a directory::

    python tools/video_sampler/main.py /data/videos -o /data/frames \\
        --clips "0:01:00-0:05:20,0:10:00-0:15:59"

Sample 2 frames/sec over the whole of a single video, deterministically::

    python tools/video_sampler/main.py clip.webm -o out --fps 2 \\
        --sampling uniform
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make `components` and `lib` importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import av  # noqa: E402

from components.clip_parser import ClipParseError, parse_clips, whole_video_clip  # noqa: E402
from components.frame_sampler import (  # noqa: E402
    SampleParams,
    estimate_frame_count,
    get_duration,
    sample_video,
)
from components.video_discovery import discover_videos  # noqa: E402
from lib.logging_utils import get_logger, setup_logging  # noqa: E402
from lib.timecode import format_timecode  # noqa: E402

logger = get_logger("video_sampler")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_sampler",
        description="Efficiently sample frames (PNG or JPEG) from videos over time clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Directory of videos (or a single video file) to sample from.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for sampled images."
    )
    parser.add_argument(
        "-c",
        "--clips",
        default=None,
        help=(
            "Comma-separated time ranges, e.g. '0:01:00-0:05:20,0:10:00-0:15:59'. "
            "Must be increasing and non-overlapping. Omit to sample the whole video."
        ),
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=1,
        help="Frames sampled per second (integer n >= 1).",
    )
    parser.add_argument(
        "-s",
        "--sampling",
        choices=("even", "random"),
        default="even",
        help="'even' keeps the frame nearest each evenly-spaced per-second target "
        "(k/(n+1) within the second; the midpoint when fps=1); "
        "'random' keeps a uniformly-random frame from each 1/n-second sub-window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible sampling in --sampling random mode.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpeg"),
        default="png",
        help="Output image format. PNG is lossless; JPEG is smaller (lossy).",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=95,
        help="JPEG quality, 0-100 (ignored for PNG, which is lossless). "
        "Values above 95 give little gain for much larger files.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Parallel videos to process at once (default: min(4, #videos)).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when scanning a directory.",
    )
    parser.add_argument(
        "--ext",
        default="mp4,webm",
        help="Comma-separated video extensions to scan for.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write all images flat into the output dir (prefixed by video name) "
        "instead of one subdirectory per video.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List videos, clips and estimated frame counts without decoding.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def _resolve_clips(spec, videos):
    if spec:
        clips = parse_clips(spec)
        logger.info(
            "Clips: %s", ", ".join(f"{c.label()}" for c in clips)
        )
    else:
        clips = [whole_video_clip()]
        logger.info("No clips given; sampling the entire duration of each video.")
    return clips


def _dry_run(videos, clips, fps) -> int:
    print(f"\nDRY RUN: {len(videos)} video(s), {len(clips)} clip(s), fps={fps}\n")
    grand_total = 0
    for video in videos:
        try:
            with av.open(str(video)) as container:
                if not container.streams.video:
                    print(f"  {video.name}: NO VIDEO STREAM")
                    continue
                stream = container.streams.video[0]
                duration = get_duration(container, stream)
        except av.FFmpegError as exc:
            print(f"  {video.name}: cannot open ({exc})")
            continue

        dur_str = format_timecode(duration) if duration is not None else "unknown"
        video_total = sum(estimate_frame_count(c, duration, fps) for c in clips)
        grand_total += video_total
        print(f"  {video.name}  (duration {dur_str})  ~{video_total} frames")
        for clip in clips:
            est = estimate_frame_count(clip, duration, fps)
            print(f"      clip {clip.index} {clip.label()}  ~{est} frames")
    print(f"\nEstimated total frames: ~{grand_total}\n")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if args.fps < 1:
        logger.error("--fps must be an integer >= 1 (got %s)", args.fps)
        return 2
    if not (0 <= args.quality <= 100):
        logger.error("--quality must be in 0..100 (got %s)", args.quality)
        return 2

    extensions = tuple(e.strip() for e in args.ext.split(",") if e.strip())

    try:
        videos = discover_videos(args.input, recursive=args.recursive, extensions=extensions)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    if not videos:
        logger.error("no video files found under %s (extensions: %s)", args.input, extensions)
        return 2

    logger.info("Found %d video(s).", len(videos))

    try:
        clips = _resolve_clips(args.clips, videos)
    except ClipParseError as exc:
        logger.error("clip error: %s", exc)
        return 2

    if args.dry_run:
        return _dry_run(videos, clips, args.fps)

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SampleParams(
        fps=args.fps,
        sampling=args.sampling,
        image_format=args.format,
        quality=args.quality,
        seed=args.seed,
        per_video_subdir=not args.flat,
    )

    workers = args.workers or min(4, len(videos))
    workers = max(1, workers)
    logger.info("Sampling with %d worker(s)...", workers)

    start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(sample_video, video, clips, out_dir, params): video
            for video in videos
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - start
    return _report(results, out_dir, elapsed)


def _report(results, out_dir: Path, elapsed: float) -> int:
    results.sort(key=lambda r: r.video.name)
    total_frames = sum(r.frames_saved for r in results)
    errors = [r for r in results if r.error]

    print("\n=== Summary ===")
    for r in results:
        status = f"ERROR: {r.error}" if r.error else f"{r.frames_saved} frames"
        print(f"  {r.video.name}: {status}")
        for warning in r.warnings:
            print(f"      warning: {warning}")
    print(
        f"\nSaved {total_frames} frame(s) from {len(results) - len(errors)}"
        f"/{len(results)} video(s) into {out_dir} in {elapsed:.1f}s"
    )
    if errors:
        print(f"{len(errors)} video(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
