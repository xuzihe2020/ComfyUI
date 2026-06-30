# video_sampler

Efficiently sample JPEG frames from long, high-resolution videos (`.mp4`, `.webm`,
and other common containers) over optional time clips.

Designed for the hard case: 1–2 hour, ~2 GB videos where decoding the whole file
would be wasteful. It uses [PyAV](https://pyav.org) (already a ComfyUI dependency,
bundles FFmpeg — no system `ffmpeg` needed) and for each clip performs **one seek**
to the keyframe at the clip start, then stream-decodes forward only through that
clip's frames.

## How sampling works

1. Each clip's time span is divided into fixed buckets of `1 / fps` seconds.
2. Exactly one frame is emitted per bucket that contains frames:
   - `--sampling random` (default): a frame chosen **uniformly at random** within
     the bucket, via reservoir sampling (one streaming pass, O(1) memory).
   - `--sampling uniform`: the first frame of the bucket (deterministic, edge-aligned).
3. Only the chosen frame is JPEG-encoded, so at most one decoded frame is held in
   memory at a time. Videos are processed in parallel across worker threads.

## Usage

```bash
# 1 frame/sec over two clips, for every video in a directory
python tools/video_sampler/main.py /data/videos -o /data/frames \
    --clips "0:01:00-0:05:20,0:10:00-0:15:59"

# 2 frames/sec over the entire duration of a single video, deterministic
python tools/video_sampler/main.py clip.webm -o out --fps 2 --sampling uniform

# Preview what would be sampled without decoding anything
python tools/video_sampler/main.py /data/videos -o out \
    --clips "0:01:00-0:05:20" --dry-run
```

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | — | Video directory, or a single video file. |
| `-o, --output` | required | Output directory for JPEGs. |
| `-c, --clips` | whole video | `start-end` ranges, comma separated. Must be increasing and non-overlapping, else it raises. Timecodes accept `SS`, `MM:SS`, `H:MM:SS`, with optional `.mmm`. |
| `-f, --fps` | `1.0` | Frames sampled per second (fractional allowed, e.g. `0.5`). |
| `-s, --sampling` | `random` | `random` or `uniform` frame within each bucket. |
| `--seed` | none | Seed for reproducible random sampling. |
| `-q, --quality` | `95` | JPEG quality, 1–100. |
| `-w, --workers` | `min(4, #videos)` | Videos processed in parallel. |
| `-r, --recursive` | off | Recurse into subdirectories. |
| `--ext` | `mp4,webm` | Extensions to scan for. |
| `--flat` | off | Write all JPEGs flat (prefixed by video name) instead of one subdir per video. |
| `--dry-run` | off | List videos/clips/estimated counts, then exit. |
| `-v, --verbose` | off | Verbose logging. |

### Output layout

By default each video gets its own subdirectory; filenames are the frame timestamp
(`HH-MM-SS.mmm.jpg`), so they sort chronologically:

```
out/
  my_long_video/
    00-01-00.123.jpg
    00-01-01.064.jpg
    ...
```

With `--flat`, files go directly in the output dir as `my_long_video_00-01-00.123.jpg`.

## Layout

```
tools/video_sampler/
  main.py                     CLI entry point and orchestration
  components/
    clip_parser.py            parse + validate clip ranges (ordering checks)
    video_discovery.py        find video files in a dir / single file
    frame_sampler.py          PyAV decode + bucket + reservoir sampling engine
  lib/
    timecode.py               timecode parse/format helpers
    logging_utils.py          logging setup
  requirements.txt
```
