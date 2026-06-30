"""Parsing and formatting of timecodes.

Accepted input forms (fractional seconds allowed everywhere):

    "SS"            -> 90      means 90 seconds
    "SS.mmm"        -> 12.5
    "MM:SS"         -> 5:20    means 320 seconds
    "H:MM:SS"       -> 0:05:20
    "HH:MM:SS.mmm"  -> 01:02:03.500
"""

from __future__ import annotations


def parse_timecode(value: str | int | float) -> float:
    """Parse a timecode into a number of seconds (float).

    Raises ValueError on anything that does not look like a valid timecode.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        raise ValueError(f"invalid timecode: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"timecode must be non-negative: {value!r}")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("empty timecode")

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' separated fields in timecode: {value!r}")

    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"non-numeric field in timecode: {value!r}") from None

    if any(n < 0 for n in numbers):
        raise ValueError(f"timecode fields must be non-negative: {value!r}")

    if len(numbers) == 1:
        seconds = numbers[0]
    elif len(numbers) == 2:
        minutes, secs = numbers
        if secs >= 60:
            raise ValueError(f"seconds field must be < 60 in {value!r}")
        seconds = minutes * 60 + secs
    else:  # len == 3
        hours, minutes, secs = numbers
        if minutes >= 60:
            raise ValueError(f"minutes field must be < 60 in {value!r}")
        if secs >= 60:
            raise ValueError(f"seconds field must be < 60 in {value!r}")
        seconds = hours * 3600 + minutes * 60 + secs

    return seconds


def format_timecode(seconds: float) -> str:
    """Format seconds as a human readable ``HH:MM:SS.mmm`` string."""
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_for_filename(seconds: float) -> str:
    """Format seconds as a filesystem-safe, lexically sortable ``HH-MM-SS.mmm``."""
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{millis:03d}"
