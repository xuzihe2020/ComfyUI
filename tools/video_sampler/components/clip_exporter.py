"""Clip export and container conversion engine (PyAV based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av

from components.frame_sampler import get_duration
from lib.logging_utils import get_logger
from lib.timecode import format_for_filename, format_timecode

logger = get_logger(__name__)

WEBM_VIDEO_CODECS = {"vp8", "vp9", "av1"}
WEBM_AUDIO_CODECS = {"opus", "vorbis"}
MP4_VIDEO_CODECS = {"h264", "hevc", "mpeg4", "av1"}
MP4_AUDIO_CODECS = {"aac", "mp3", "opus"}

VIDEO_CODEC_MAP = {
    "vp9": "libvpx-vp9",
    "vp8": "libvpx",
    "av1": "libaom-av1",
    "h264": "libx264",
    "hevc": "libx265",
    "mpeg4": "mpeg4",
}
AUDIO_CODEC_MAP = {
    "opus": "libopus",
    "vorbis": "vorbis",
    "aac": "aac",
    "mp3": "libmp3lame",
}


@dataclass
class ClipParams:
    output_format: str = "webm"
    video_codec: str | None = None
    audio_codec: str | None = None
    fps: float | None = None
    video_crf: int | None = None
    video_bitrate: str | None = None
    audio_bitrate: str | None = None
    preset: str | None = None
    per_video_subdir: bool = True


@dataclass
class ClipExportResult:
    video: Path
    duration: float | None
    clips_saved: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def export_video_clips(
    video: Path,
    clips,
    out_dir: Path,
    params: ClipParams,
) -> ClipExportResult:
    """Export every clip of one video. Never raises; failures land in ``result.error``."""
    video = Path(video)
    result = ClipExportResult(video=video, duration=None)

    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                result.error = "no video stream"
                return result

            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            audio_stream = container.streams.audio[0] if container.streams.audio else None
            duration = get_duration(container, video_stream)
            result.duration = duration

            target_dir = out_dir / video.stem if params.per_video_subdir else out_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            prefix = "" if params.per_video_subdir else f"{video.stem}_"

            for clip in clips:
                if not _clip_has_video_span(clip, duration, result):
                    continue
                out_path = _clip_output_path(
                    target_dir,
                    prefix,
                    clip,
                    params.output_format,
                )
                _export_clip(
                    video,
                    clip,
                    duration,
                    out_path,
                    params,
                    video_stream,
                    audio_stream,
                    result,
                )
                result.clips_saved += 1
                logger.info(
                    "%s clip %d %s -> %s",
                    video.name,
                    clip.index,
                    clip.label(),
                    out_path,
                )
    except av.FFmpegError as exc:
        result.error = f"encode error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _clip_has_video_span(clip, duration: float | None, result: ClipExportResult) -> bool:
    if duration is None:
        return True
    if clip.start >= duration:
        msg = (
            f"clip {clip.index} {clip.label()} starts at/after video duration "
            f"({format_timecode(duration)}); skipped"
        )
        logger.warning("%s: %s", result.video.name, msg)
        result.warnings.append(msg)
        return False
    if clip.end is not None and clip.end > duration:
        msg = (
            f"clip {clip.index} end clamped from {format_timecode(clip.end)} "
            f"to video duration {format_timecode(duration)}"
        )
        logger.warning("%s: %s", result.video.name, msg)
        result.warnings.append(msg)
    return True


def _clip_output_path(target_dir: Path, prefix: str, clip, output_format: str) -> Path:
    start = format_for_filename(clip.start)
    end = "end" if clip.end is None else format_for_filename(clip.end)
    return target_dir / f"{prefix}clip_{clip.index:03d}_{start}_{end}.{output_format}"


def _export_clip(
    source_path: Path,
    clip,
    duration: float | None,
    out_path: Path,
    params: ClipParams,
    source_video_stream,
    source_audio_stream,
    result: ClipExportResult,
) -> None:
    clip_start = clip.start
    clip_end = clip.end
    if duration is not None and clip_end is not None:
        clip_end = min(clip_end, duration)

    video_codec = _resolve_video_codec(
        source_path,
        params.output_format,
        source_video_stream,
        params.video_codec,
    )
    audio_codec = None
    if source_audio_stream is not None:
        audio_codec = _resolve_audio_codec(
            source_path,
            params.output_format,
            source_audio_stream,
            params.audio_codec,
        )

    with av.open(str(source_path)) as input_container, av.open(
        str(out_path),
        mode="w",
        format=params.output_format,
    ) as output_container:
        in_video = input_container.streams.video[0]
        in_video.thread_type = "AUTO"
        in_audio = input_container.streams.audio[0] if input_container.streams.audio else None

        output_fps = _resolve_fps(in_video, params.fps)
        out_video = output_container.add_stream(video_codec, rate=output_fps)
        out_video.width = in_video.codec_context.width
        out_video.height = in_video.codec_context.height
        out_video.pix_fmt = "yuv420p"
        out_video.time_base = Fraction(output_fps.denominator, output_fps.numerator)
        _apply_video_options(out_video, params, params.output_format)

        out_audio = None
        if in_audio is not None and audio_codec is not None:
            out_audio = output_container.add_stream(
                audio_codec,
                rate=in_audio.codec_context.rate or 48000,
            )
            out_audio.layout = in_audio.codec_context.layout.name if in_audio.codec_context.layout else "stereo"
            if params.audio_bitrate:
                out_audio.bit_rate = _parse_bitrate(params.audio_bitrate)

        if clip_start > 0:
            seek_stream = in_video
            seek_target = int(clip_start / float(seek_stream.time_base))
            input_container.seek(
                seek_target,
                stream=seek_stream,
                backward=True,
                any_frame=False,
            )

        video_done = False
        audio_done = out_audio is None
        streams = [in_video]
        if in_audio is not None and out_audio is not None:
            streams.append(in_audio)

        video_sampler = _ClipFrameSelector(clip_start, clip_end, params.fps)

        for packet in input_container.demux(streams):
            if packet.stream == in_video and not video_done:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    t = frame.pts * float(in_video.time_base)
                    if t < clip_start:
                        continue
                    if clip_end is not None and t >= clip_end:
                        video_done = True
                        break
                    for selected_frame in video_sampler.select(frame, t):
                        _encode_video_frame(selected_frame, out_video, output_container)
            elif packet.stream == in_audio and not audio_done and out_audio is not None:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    t = frame.pts * float(in_audio.time_base)
                    if t < clip_start:
                        continue
                    if clip_end is not None and t >= clip_end:
                        audio_done = True
                        break
                    frame.pts = None
                    for out_packet in out_audio.encode(frame):
                        output_container.mux(out_packet)

            if video_done and audio_done:
                break

        for selected_frame in video_sampler.finish():
            _encode_video_frame(selected_frame, out_video, output_container)

        for out_packet in out_video.encode():
            output_container.mux(out_packet)
        if out_audio is not None:
            for out_packet in out_audio.encode():
                output_container.mux(out_packet)


class _ClipFrameSelector:
    def __init__(self, clip_start: float, clip_end: float | None, fps: float | None) -> None:
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.step = None if fps is None else 1.0 / fps
        self.next_target = clip_start
        self.prev_frame = None
        self.prev_t = 0.0

    def select(self, frame, t: float):
        if self.step is None:
            return [frame]

        selected = []
        while self._target_in_range(self.next_target) and t >= self.next_target:
            if self.prev_frame is not None and abs(self.prev_t - self.next_target) <= abs(
                t - self.next_target
            ):
                selected.append(self.prev_frame)
            else:
                selected.append(frame)
            self.next_target += self.step

        self.prev_frame = frame
        self.prev_t = t
        return selected

    def finish(self):
        if self.step is None or self.clip_end is None or self.prev_frame is None:
            return []

        selected = []
        while self.next_target < self.clip_end:
            selected.append(self.prev_frame)
            self.next_target += self.step
        return selected

    def _target_in_range(self, target: float) -> bool:
        return self.clip_end is None or target < self.clip_end


def _encode_video_frame(frame, stream, container) -> None:
    frame = frame.reformat(width=stream.width, height=stream.height, format="yuv420p")
    frame.pts = None
    for out_packet in stream.encode(frame):
        container.mux(out_packet)


def _resolve_fps(stream, requested_fps: float | None):
    if requested_fps is not None:
        return Fraction(requested_fps).limit_denominator(1001)
    if stream.average_rate is not None:
        return stream.average_rate
    if stream.base_rate is not None:
        return stream.base_rate
    return Fraction(30, 1)


def _resolve_video_codec(source_path: Path, output_format: str, stream, requested: str | None) -> str:
    codec = requested or _default_video_codec(source_path, output_format, stream)
    return VIDEO_CODEC_MAP.get(codec, codec)


def _resolve_audio_codec(source_path: Path, output_format: str, stream, requested: str | None) -> str:
    codec = requested or _default_audio_codec(source_path, output_format, stream)
    return AUDIO_CODEC_MAP.get(codec, codec)


def _default_video_codec(source_path: Path, output_format: str, stream) -> str:
    source_codec = stream.codec_context.name
    if output_format == "webm":
        if (
            source_path.suffix.lower() == ".webm"
            and source_codec in WEBM_VIDEO_CODECS
            and _encoder_available(source_codec, VIDEO_CODEC_MAP)
        ):
            return source_codec
        return "vp9"
    if (
        source_path.suffix.lower() == ".mp4"
        and source_codec in MP4_VIDEO_CODECS
        and _encoder_available(source_codec, VIDEO_CODEC_MAP)
    ):
        return source_codec
    return "h264"


def _default_audio_codec(source_path: Path, output_format: str, stream) -> str:
    source_codec = stream.codec_context.name
    if output_format == "webm":
        if (
            source_path.suffix.lower() == ".webm"
            and source_codec in WEBM_AUDIO_CODECS
            and _encoder_available(source_codec, AUDIO_CODEC_MAP)
        ):
            return source_codec
        return "opus"
    if (
        source_path.suffix.lower() == ".mp4"
        and source_codec in MP4_AUDIO_CODECS
        and _encoder_available(source_codec, AUDIO_CODEC_MAP)
    ):
        return source_codec
    return "aac"


def _encoder_available(codec: str, codec_map: dict[str, str]) -> bool:
    try:
        av.codec.Codec(codec_map.get(codec, codec), "w")
    except Exception:
        return False
    return True


def _apply_video_options(stream, params: ClipParams, output_format: str) -> None:
    if params.video_bitrate:
        stream.bit_rate = _parse_bitrate(params.video_bitrate)

    options = {}
    if params.video_crf is not None:
        options["crf"] = str(params.video_crf)
        if output_format == "webm" and not params.video_bitrate:
            options["b:v"] = "0"
    if params.preset:
        options["preset"] = params.preset
    if options:
        stream.codec_context.options = options


def _parse_bitrate(value: str) -> int:
    cleaned = value.strip().lower()
    multipliers = {"k": 1_000, "m": 1_000_000}
    suffix = cleaned[-1]
    if suffix in multipliers:
        return int(float(cleaned[:-1]) * multipliers[suffix])
    return int(cleaned)
