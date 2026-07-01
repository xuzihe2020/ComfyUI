# video_sampler

Efficiently sample frames (PNG or JPEG) from long, high-resolution videos
(`.mp4`, `.webm`, and other common containers) over optional time clips.
Output images default to the source frame's native resolution. Use `--scale` to
shrink outputs while preserving aspect ratio.

Designed for the hard case: 1–2 hour, ~2 GB videos where decoding the whole file
would be wasteful. It uses [PyAV](https://pyav.org) (already a ComfyUI dependency,
bundles FFmpeg — no system `ffmpeg` needed) and for each clip performs **one seek**
to the keyframe at the clip start, then stream-decodes forward only through that
clip's frames.

## How sampling works

`--fps` is an integer `n` = frames kept per second. Two modes choose *which*
frame(s) in each second to keep:

- **`--sampling even` (default):** divide each 1-second window into `n + 1` equal
  chunks and take the `n` interior boundary points as target times —
  `k / (n + 1)` for `k = 1..n`. So `fps=1` → the **midpoint** (0.5s); `fps=2` →
  `1/3, 2/3`; `fps=3` → `1/4, 2/4, 3/4`. The decoded frame **nearest each target**
  is kept (deterministic, evenly distributed).
- **`--sampling random`:** split each second into `n` equal sub-windows and keep
  one frame chosen **uniformly at random** from each, via reservoir sampling.

Either way exactly one frame is encoded (PNG or JPEG) per target, so at most one
decoded frame is held in memory at a time. Per clip, the engine seeks once to the keyframe
at the clip start then stream-decodes only that clip. Videos are processed in
parallel across worker threads.

## Usage

```bash
# 1 frame/sec over two clips, for every video in a directory
python tools/video_sampler/main.py /data/videos -o /data/frames \
    --clips "0:01:00-0:05:20,0:10:00-0:15:59"

# 2 frames/sec (at 1/3 and 2/3 of each second) over the whole of a single video
python tools/video_sampler/main.py clip.webm -o out --fps 2

# 1 random frame/sec, output smaller JPEGs instead of the default lossless PNG
python tools/video_sampler/main.py clip.webm -o out --sampling random --seed 7 \
    --format jpeg --quality 90

# Shrink output frames to 80% of the source resolution
python tools/video_sampler/main.py clip.webm -o out --scale 0.8

# Preview what would be sampled without decoding anything
python tools/video_sampler/main.py /data/videos -o out \
    --clips "0:01:00-0:05:20" --dry-run
```

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | — | Video directory, or a single video file. |
| `-o, --output` | required | Output directory for images. |
| `-c, --clips` | whole video | `start-end` ranges, comma separated. Must be increasing and non-overlapping, else it raises. Timecodes accept `SS`, `MM:SS`, `H:MM:SS`, with optional `.mmm`. |
| `-f, --fps` | `1` | Frames sampled per second (integer `n >= 1`). |
| `-s, --sampling` | `even` | `even` (frame nearest each `k/(n+1)` target) or `random` (random frame per `1/n` sub-window). |
| `--seed` | none | Seed for reproducible `random` sampling. |
| `--format` | `png` | Output image format: `png` (lossless) or `jpeg` (lossy, smaller). |
| `-q, --quality` | `95` | JPEG quality, 0–100. Ignored for PNG. Values >95 bloat the file for little gain. |
| `--scale` | `1.0` | Output image scale, `0.0 < scale <= 1.0`. Preserves aspect ratio; `1.0` keeps native resolution, `0.8` turns `1920x1280` into `1536x1024`. |
| `-w, --workers` | `min(4, #videos)` | Videos processed in parallel. |
| `-r, --recursive` | off | Recurse into subdirectories. |
| `--ext` | `mp4,webm` | Extensions to scan for. |
| `--flat` | off | Write all images flat (prefixed by video name) instead of one subdir per video. |
| `--dry-run` | off | List videos/clips/estimated counts, then exit. |
| `-v, --verbose` | off | Verbose logging. |

### Output layout

By default each video gets its own subdirectory; filenames are the frame timestamp
(`HH-MM-SS.mmm.<ext>`, `.png` or `.jpg`), so they sort chronologically:

```
out/
  my_long_video/
    00-01-00.123.png
    00-01-01.064.png
    ...
```

With `--flat`, files go directly in the output dir as `my_long_video_00-01-00.123.png`.

## Layout

```
tools/video_sampler/
  main.py                     CLI entry point and orchestration
  components/
    clip_parser.py            parse + validate clip ranges (ordering checks)
    video_discovery.py        find video files in a dir / single file
    frame_sampler.py          PyAV decode + even/random per-second sampling engine
  lib/
    timecode.py               timecode parse/format helpers
    logging_utils.py          logging setup
  requirements.txt
```
