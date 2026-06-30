"""The frame sampling engine (PyAV based).

Sampling strategy, optimized for long high-resolution videos:

* Open each video once. For every clip, seek a single time to the keyframe at or
  before the clip start, then stream-decode forward only through that clip's
  frames. We never decode the whole file and never seek per output frame.
* Frames are grouped into fixed-width time buckets of ``1 / fps`` seconds. Exactly
  one frame is emitted per bucket that contains frames:
    - ``random``  : reservoir-sample one frame uniformly at random from the bucket.
    - ``uniform`` : keep the first frame of the bucket (closest to the bucket edge).
* The chosen frame is encoded to JPEG only when its bucket closes, so at most one
  decoded frame is held in memory at a time.
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
    fps: float = 1.0
    sampling: str = "random"  # "random" | "uniform"
    quality: int = 95
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

    bucket_dur = 1.0 / params.fps
    time_base = float(stream.time_base)

    # Seek to the keyframe at or before the clip start (input seeking on the stream).
    if clip_start > 0:
        seek_target = int(clip_start / time_base)
        container.seek(seek_target, stream=stream, backward=True, any_frame=False)

    current_bucket: int | None = None
    chosen_frame = None
    chosen_time = 0.0
    seen_in_bucket = 0
    saved = 0

    def flush() -> int:
        if chosen_frame is None:
            return 0
        _save_frame(chosen_frame, chosen_time, target_dir, prefix, params.quality)
        return 1

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
            saved += flush()
            current_bucket = bucket
            chosen_frame = None
            seen_in_bucket = 0

        if params.sampling == "uniform":
            if chosen_frame is None:
                chosen_frame = frame
                chosen_time = t
        else:  # reservoir sampling, k = 1
            seen_in_bucket += 1
            if rng.random() < 1.0 / seen_in_bucket:
                chosen_frame = frame
                chosen_time = t

    saved += flush()
    return saved


def _save_frame(frame, t: float, target_dir: Path, prefix: str, quality: int) -> None:
    image = frame.to_image()  # PIL.Image in RGB
    out_path = target_dir / f"{prefix}{format_for_filename(t)}.jpg"
    image.save(out_path, format="JPEG", quality=quality)


def estimate_frame_count(clip, duration: float | None, fps: float) -> int:
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
