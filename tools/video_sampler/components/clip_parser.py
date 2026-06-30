"""Parsing and validation of clip range specifications.

A clip spec is a list of ``start-end`` ranges expressed as timecodes, e.g.::

    0:01:00-0:05:20,0:10:00-0:15:59

Clips must be strictly increasing and non-overlapping: each clip's start must be
>= the previous clip's end. Anything else raises ``ClipParseError``.
"""

from __future__ import annotations

from dataclasses import dataclass

from lib.timecode import format_timecode, parse_timecode


class ClipParseError(ValueError):
    """Raised when a clip specification is malformed or out of order."""


@dataclass(frozen=True)
class Clip:
    """A half-open time range ``[start, end)`` in seconds."""

    start: float
    end: float | None  # None means "until the end of the video"
    index: int

    @property
    def duration(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start

    def label(self) -> str:
        end = "end" if self.end is None else format_timecode(self.end)
        return f"[{format_timecode(self.start)} - {end}]"


def parse_clips(spec: str | list[str]) -> list[Clip]:
    """Parse a clip spec into validated, ordered :class:`Clip` objects.

    ``spec`` may be a single comma-separated string or a list of ``start-end``
    tokens. Whitespace and surrounding brackets are tolerated.
    """
    if isinstance(spec, str):
        cleaned = spec.strip().lstrip("[").rstrip("]")
        tokens = [t.strip() for t in cleaned.split(",")]
    else:
        tokens = [str(t).strip() for t in spec]

    tokens = [t for t in tokens if t]
    if not tokens:
        raise ClipParseError("no clips found in specification")

    clips: list[Clip] = []
    for raw in tokens:
        start, end = _parse_range(raw)
        index = len(clips)
        if clips:
            previous = clips[-1]
            # previous.end is never None here: only the implicit whole-video clip
            # uses None, and that path never goes through parse_clips.
            if start < previous.end:
                raise ClipParseError(
                    f"clips must be in increasing, non-overlapping order: "
                    f"clip {index} starts at {format_timecode(start)} which is "
                    f"before the end of clip {index - 1} "
                    f"({format_timecode(previous.end)})"
                )
        clips.append(Clip(start=start, end=end, index=index))

    return clips


def _parse_range(raw: str) -> tuple[float, float]:
    if "-" not in raw:
        raise ClipParseError(
            f"clip {raw!r} is not a range; expected 'start-end' (e.g. 0:01:00-0:05:20)"
        )
    start_text, _, end_text = raw.partition("-")
    try:
        start = parse_timecode(start_text)
        end = parse_timecode(end_text)
    except ValueError as exc:
        raise ClipParseError(f"bad timecode in clip {raw!r}: {exc}") from exc

    if end <= start:
        raise ClipParseError(
            f"clip {raw!r} has end ({format_timecode(end)}) <= start "
            f"({format_timecode(start)})"
        )
    return start, end


def whole_video_clip() -> Clip:
    """The implicit clip used when the caller supplies no ranges."""
    return Clip(start=0.0, end=None, index=0)
