#!/usr/bin/env python3
"""Sample frames or export video clips over optional time ranges.

Examples
--------
Sample 60 frames/minute over two ranges from every video in a directory::

    python tools/video_sampler/main.py /data/videos -o /data/frames \\
        --clips "0:01:00-0:05:20,0:10:00-0:15:59"

Sample 12 frames/minute over the whole of a single video::

    python tools/video_sampler/main.py clip.webm -o out --frames-per-minute 12

Export WebM clips from the same range syntax::

    python tools/video_sampler/main.py clip.mp4 -o out --mode clip \\
        --clips "0:01:00-0:05:20"
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

from components.clip_exporter import ClipParams, export_video_clips  # noqa: E402
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
        description="Efficiently sample frames or export video clips over time ranges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Directory of videos, or a single video file, to process.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for frames or clips."
    )
    parser.add_argument(
        "--mode",
        choices=("sample", "clip"),
        default="sample",
        help="Run mode: sample still frames or export video clips.",
    )
    parser.add_argument(
        "-c",
        "--clips",
        default=None,
        help=(
            "Comma-separated time ranges, e.g. '0:01:00-0:05:20,0:10:00-0:15:59'. "
            "Must be increasing and non-overlapping. Omit to process the whole video."
        ),
    )
    parser.add_argument(
        "-f",
        "--frames-per-minute",
        type=int,
        default=60,
        help="Sample mode only: frames kept per minute (integer n >= 1).",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpeg"),
        default="png",
        help="Sample mode only: output image format. PNG is lossless; JPEG is smaller.",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=95,
        help="Sample mode only: JPEG quality, 0-100 (ignored for PNG). "
        "Values above 95 give little gain for much larger files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Sample mode only: output image scale, 0.0 < scale <= 1.0. "
        "Use 1.0 to keep native resolution; e.g. 0.8 shrinks 1920x1280 to 1536x1024.",
    )
    parser.add_argument(
        "--video-format",
        choices=("webm", "mp4"),
        default="webm",
        help="Clip mode only: output container format.",
    )
    parser.add_argument(
        "--clip-fps",
        type=float,
        default=None,
        help="Clip mode only: output video FPS. Omit to keep the input video's FPS.",
    )
    parser.add_argument(
        "--video-codec",
        choices=("vp9", "vp8", "h264", "hevc", "mpeg4"),
        default=None,
        help="Clip mode only: output video codec. Omit for input-compatible defaults.",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("opus", "vorbis", "aac", "mp3"),
        default=None,
        help="Clip mode only: output audio codec. Omit for input-compatible defaults.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=None,
        help="Clip mode only: video CRF/quality value passed to the encoder.",
    )
    parser.add_argument(
        "--video-bitrate",
        default=None,
        help="Clip mode only: video bitrate, e.g. 2500k or 4m.",
    )
    parser.add_argument(
        "--audio-bitrate",
        default=None,
        help="Clip mode only: audio bitrate, e.g. 128k.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Clip mode only: encoder preset when supported by the selected codec.",
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
        help="Write all outputs flat into the output dir (prefixed by video name) "
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


def _dry_run(videos, clips, args) -> int:
    if args.mode == "sample":
        print(
            f"\nDRY RUN: {len(videos)} video(s), {len(clips)} clip(s), "
            f"frames_per_minute={args.frames_per_minute}\n"
        )
    else:
        print(
            f"\nDRY RUN: {len(videos)} video(s), {len(clips)} clip(s), "
            f"clip format={args.video_format}\n"
        )
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
        if args.mode == "sample":
            video_total = sum(
                estimate_frame_count(c, duration, args.frames_per_minute)
                for c in clips
            )
        else:
            video_total = sum(1 for c in clips if duration is None or c.start < duration)
        grand_total += video_total
        unit = "frames" if args.mode == "sample" else "clips"
        print(f"  {video.name}  (duration {dur_str})  ~{video_total} {unit}")
        for clip in clips:
            if args.mode == "sample":
                est = estimate_frame_count(clip, duration, args.frames_per_minute)
                print(f"      clip {clip.index} {clip.label()}  ~{est} frames")
            else:
                status = "skipped" if duration is not None and clip.start >= duration else "1 clip"
                print(f"      clip {clip.index} {clip.label()}  {status}")
    print(f"\nEstimated total {unit}: ~{grand_total}\n")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if args.frames_per_minute < 1:
        logger.error(
            "--frames-per-minute must be an integer >= 1 (got %s)",
            args.frames_per_minute,
        )
        return 2
    if not (0 <= args.quality <= 100):
        logger.error("--quality must be in 0..100 (got %s)", args.quality)
        return 2
    if not (0.0 < args.scale <= 1.0):
        logger.error("--scale must be in (0.0, 1.0] (got %s)", args.scale)
        return 2
    if args.clip_fps is not None and args.clip_fps <= 0:
        logger.error("--clip-fps must be > 0 (got %s)", args.clip_fps)
        return 2
    if args.video_crf is not None and not (0 <= args.video_crf <= 63):
        logger.error("--video-crf must be in 0..63 (got %s)", args.video_crf)
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
        return _dry_run(videos, clips, args)

    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        params = SampleParams(
            frames_per_minute=args.frames_per_minute,
            image_format=args.format,
            quality=args.quality,
            scale=args.scale,
            per_video_subdir=not args.flat,
        )
        worker_fn = sample_video
        report_fn = _report_sample_results
        action = "Sampling"
    else:
        params = ClipParams(
            output_format=args.video_format,
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            fps=args.clip_fps,
            video_crf=args.video_crf,
            video_bitrate=args.video_bitrate,
            audio_bitrate=args.audio_bitrate,
            preset=args.preset,
            per_video_subdir=not args.flat,
        )
        worker_fn = export_video_clips
        report_fn = _report_clip_results
        action = "Exporting clips"

    workers = args.workers or min(4, len(videos))
    workers = max(1, workers)
    logger.info("%s with %d worker(s)...", action, workers)

    start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker_fn, video, clips, out_dir, params): video
            for video in videos
        }
        for future in as_completed(futures):
            results.append(future.result())

    elapsed = time.monotonic() - start
    return report_fn(results, out_dir, elapsed)


def _report_sample_results(results, out_dir: Path, elapsed: float) -> int:
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


def _report_clip_results(results, out_dir: Path, elapsed: float) -> int:
    results.sort(key=lambda r: r.video.name)
    total_clips = sum(r.clips_saved for r in results)
    errors = [r for r in results if r.error]

    print("\n=== Summary ===")
    for r in results:
        status = f"ERROR: {r.error}" if r.error else f"{r.clips_saved} clips"
        print(f"  {r.video.name}: {status}")
        for warning in r.warnings:
            print(f"      warning: {warning}")
    print(
        f"\nSaved {total_clips} clip(s) from {len(results) - len(errors)}"
        f"/{len(results)} video(s) into {out_dir} in {elapsed:.1f}s"
    )
    if errors:
        print(f"{len(errors)} video(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
