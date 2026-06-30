"""The frame sampling engine (PyAV based).

Optimized for long high-resolution videos: open each video once, and for every
clip seek a single time to the keyframe at or before the clip start, then
stream-decode forward only through that clip's frames. We never decode the whole
file and never seek per output frame.

Two sampling modes select which frame(s) to keep per second (``fps`` = n frames
per second, an integer):

* ``even`` (default): divide each 1-second window into ``n + 1`` equal chunks and
  take the ``n`` interior boundary points as targets, i.e. times ``k / (n + 1)``
  for ``k = 1..n`` within the second. For ``n = 1`` that is the midpoint (0.5s).
  The decoded frame nearest each target time is kept.
* ``random``: split each second into ``n`` equal sub-windows and keep one frame
  chosen uniformly at random from each, via reservoir sampling (single pass,
  O(1) memory).

Either way exactly one frame is encoded (PNG or JPEG) per emitted target, so at
most one decoded frame is held in memory at a time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import av

from lib.logging_utils import get_logger
from lib.timecode import format_for_filename, format_timecode

logger = get_logger(__name__)


@dataclass
class SampleParams:
    fps: int = 1
    sampling: str = "even"  # "even" | "random"
    image_format: str = "png"  # "png" (lossless) | "jpeg" (lossy)
    quality: int = 95  # JPEG only; ignored for PNG
    seed: int | None = None
    per_video_subdir: bool = True


@dataclass
class VideoResult:
    video: Path
    duration: float | None
    frames_saved: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def get_duration(container: "av.container.InputContainer", stream) -> float | None:
    """Best-effort duration of the video stream, in seconds."""
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def sample_video(
    video: Path,
    clips,
    out_dir: Path,
    params: SampleParams,
) -> VideoResult:
    """Sample every clip of one video. Never raises; failures land in ``result.error``."""
    video = Path(video)
    result = VideoResult(video=video, duration=None)
    rng = random.Random(params.seed)

    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                result.error = "no video stream"
                return result
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # multithreaded decode

            duration = get_duration(container, stream)
            result.duration = duration

            target_dir = out_dir / video.stem if params.per_video_subdir else out_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            prefix = "" if params.per_video_subdir else f"{video.stem}_"

            for clip in clips:
                saved = _sample_clip(
                    container,
                    stream,
                    clip,
                    duration,
                    target_dir,
                    prefix,
                    params,
                    rng,
                    result,
                )
                result.frames_saved += saved
                logger.info(
                    "%s clip %d %s -> %d frames",
                    video.name,
                    clip.index,
                    clip.label(),
                    saved,
                )
    except av.FFmpegError as exc:
        result.error = f"decode error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _sample_clip(
    container,
    stream,
    clip,
    duration: float | None,
    target_dir: Path,
    prefix: str,
    params: SampleParams,
    rng: random.Random,
    result: VideoResult,
) -> int:
    clip_start = clip.start
    clip_end = clip.end

    if duration is not None:
        if clip_start >= duration:
            msg = (
                f"clip {clip.index} {clip.label()} starts at/after video duration "
                f"({format_timecode(duration)}); skipped"
            )
            logger.warning("%s: %s", result.video.name, msg)
            result.warnings.append(msg)
            return 0
        if clip_end is not None and clip_end > duration:
            msg = (
                f"clip {clip.index} end clamped from {format_timecode(clip_end)} "
                f"to video duration {format_timecode(duration)}"
            )
            logger.warning("%s: %s", result.video.name, msg)
            result.warnings.append(msg)
            clip_end = duration

    n = params.fps
    time_base = float(stream.time_base)

    # Seek to the keyframe at or before the clip start (input seeking on the stream).
    if clip_start > 0:
        seek_target = int(clip_start / time_base)
        container.seek(seek_target, stream=stream, backward=True, any_frame=False)

    if params.image_format == "jpeg":
        ext = ".jpg"
        save_kwargs = {"format": "JPEG", "quality": params.quality}
    else:
        ext = ".png"
        save_kwargs = {"format": "PNG"}

    # Dedupe by output filename so two targets that resolve to the same frame
    # (e.g. a sub-second clip tail) never overwrite each other.
    seen_keys: set[str] = set()

    def save(frame, t: float) -> int:
        key = format_for_filename(t)
        if key in seen_keys:
            return 0
        seen_keys.add(key)
        out_path = target_dir / f"{prefix}{key}{ext}"
        frame.to_image().save(out_path, **save_kwargs)
        return 1

    if params.sampling == "random":
        return _collect_random(container, stream, clip_start, clip_end, n, time_base, rng, save)
    return _collect_even(container, stream, clip_start, clip_end, n, time_base, save)


def _collect_random(
    container, stream, clip_start, clip_end, n, time_base, rng, save
) -> int:
    """One uniformly-random frame per 1/n-second sub-window (reservoir, k=1)."""
    bucket_dur = 1.0 / n
    current_bucket = None
    chosen_frame = None
    chosen_time = 0.0
    seen_in_bucket = 0
    saved = 0

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t = frame.pts * time_base
        if t < clip_start:
            continue
        if clip_end is not None and t >= clip_end:
            break

        bucket = int((t - clip_start) / bucket_dur)
        if bucket != current_bucket:
            if chosen_frame is not None:
                saved += save(chosen_frame, chosen_time)
            current_bucket = bucket
            chosen_frame = None
            seen_in_bucket = 0

        seen_in_bucket += 1
        if rng.random() < 1.0 / seen_in_bucket:
            chosen_frame = frame
            chosen_time = t

    if chosen_frame is not None:
        saved += save(chosen_frame, chosen_time)
    return saved


def _collect_even(container, stream, clip_start, clip_end, n, time_base, save) -> int:
    """Keep the frame nearest each evenly-spaced per-second target time.

    Targets per second i (relative to clip start): clip_start + i + k/(n+1),
    for k = 1..n. Generated lazily so an open-ended clip (unknown duration) works.
    """
    sec_i = 0
    k = 1

    def next_target() -> float:
        nonlocal sec_i, k
        g = clip_start + sec_i + k / (n + 1)
        k += 1
        if k > n:
            k = 1
            sec_i += 1
        return g

    def target_in_range(g: float) -> bool:
        return clip_end is None or g < clip_end

    cur_target = next_target()
    prev_frame = None
    prev_t = 0.0
    saved = 0

    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        t = frame.pts * time_base
        if t < clip_start:
            continue
        if clip_end is not None and t >= clip_end:
            break

        # Assign every target we have now passed to its nearest decoded frame.
        while target_in_range(cur_target) and t >= cur_target:
            if prev_frame is not None and abs(prev_t - cur_target) <= abs(t - cur_target):
                saved += save(prev_frame, prev_t)
            else:
                saved += save(frame, t)
            cur_target = next_target()

        prev_frame = frame
        prev_t = t

    # Targets that fall between the last decoded frame and a known clip end map
    # to that last frame (bounded by clip_end, so this terminates).
    if clip_end is not None and prev_frame is not None:
        while cur_target < clip_end:
            saved += save(prev_frame, prev_t)
            cur_target = next_target()

    # Guarantee a non-empty clip yields at least one frame (e.g. sub-second clips
    # whose only target fell outside the clip bounds).
    if saved == 0 and prev_frame is not None:
        saved += save(prev_frame, prev_t)

    return saved


def estimate_frame_count(clip, duration: float | None, fps: int) -> int:
    """Upper-bound estimate of frames a clip will yield (for --dry-run)."""
    start = clip.start
    end = clip.end if clip.end is not None else duration
    if end is None:
        return 0  # unknown duration, cannot estimate
    if duration is not None:
        end = min(end, duration)
        if start >= duration:
            return 0
    span = max(0.0, end - start)
    return int(math.ceil(span * fps))
