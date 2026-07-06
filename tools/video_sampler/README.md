# video_sampler

Efficiently sample still frames (PNG or JPEG) or export video clips from long,
high-resolution videos (`.mp4`, `.webm`, and other common containers) over
optional time ranges. Frame outputs default to the source frame's native
resolution. Use `--scale` in sample mode to shrink outputs while preserving
aspect ratio.

Designed for the hard case: 1-2 hour, ~2 GB videos where decoding the whole file
would be wasteful. The tool uses [PyAV](https://pyav.org) (already a ComfyUI
dependency, bundles FFmpeg, no system `ffmpeg` needed). For each range it seeks
once to the keyframe near the clip start, then decodes forward only through that
range.

## Modes

### Sample Mode

`--mode sample` is the default. It writes uniformly spaced still frames.

`--frames-per-minute` is an integer `n` = frames kept per minute:

- `5` keeps one frame every 12 seconds.
- `12` keeps one frame every 5 seconds.
- `60` keeps one frame every second.
- `120` keeps two frames every second.

The decoded frame nearest each uniform target time is kept. Random sampling has
been removed.

### Clip Mode

`--mode clip` writes video clips for the same `--clips` ranges. The default
output container is WebM. Use `--video-format mp4` to write MP4 instead.

When codec args are omitted, the tool keeps input-compatible defaults where
practical:

- WebM input to WebM output keeps the input WebM video/audio codec when it is
  WebM-compatible.
- MP4 input to default WebM output uses VP9 video and Opus audio.
- MP4 output keeps compatible MP4 codecs for MP4 input, otherwise H.264/AAC.

Use `--video-codec vp9|vp8|h264|hevc|mpeg4` and
`--audio-codec opus|vorbis|aac|mp3` to override those defaults. `--clip-fps`
sets output FPS; omit it to keep the input video's FPS. Optional encoder knobs
include `--video-crf`, `--video-bitrate`, `--audio-bitrate`, and `--preset`.

## Usage

```bash
# 60 frames/minute over two ranges, for every video in a directory
python tools/video_sampler/main.py /data/videos -o /data/frames \
    --clips "0:01:00-0:05:20,0:10:00-0:15:59"

# One frame every 5 seconds over the whole of a single video
python tools/video_sampler/main.py clip.webm -o out --frames-per-minute 12

# Two frames per second, output smaller JPEGs
python tools/video_sampler/main.py clip.webm -o out --frames-per-minute 120 \
    --format jpeg --quality 90

# Shrink sampled frames to 80% of the source resolution
python tools/video_sampler/main.py clip.webm -o out --scale 0.8

# Export ranged WebM clips from an MP4 input using default VP9/Opus
python tools/video_sampler/main.py clip.mp4 -o out --mode clip \
    --clips "0:01:00-0:05:20"

# Convert ranged WebM clips to MP4 and set FPS/quality
python tools/video_sampler/main.py clip.webm -o out --mode clip \
    --video-format mp4 --clip-fps 24 --video-crf 23

# Preview what would be processed without decoding anything
python tools/video_sampler/main.py /data/videos -o out \
    --clips "0:01:00-0:05:20" --dry-run
```

### Arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | - | Video directory, or a single video file. |
| `-o, --output` | required | Output directory for frames or clips. |
| `--mode` | `sample` | `sample` writes still frames; `clip` writes video clips. |
| `-c, --clips` | whole video | `start-end` ranges, comma separated. Must be increasing and non-overlapping. Timecodes accept `SS`, `MM:SS`, `H:MM:SS`, with optional `.mmm`. |
| `-f, --frames-per-minute` | `60` | Sample mode only. Uniform frames sampled per minute, integer `n >= 1`. |
| `--format` | `png` | Sample mode only. Output image format: `png` or `jpeg`. |
| `-q, --quality` | `95` | Sample mode only. JPEG quality, 0-100. Ignored for PNG. |
| `--scale` | `1.0` | Sample mode only. Output image scale, `0.0 < scale <= 1.0`. |
| `--video-format` | `webm` | Clip mode only. Output container: `webm` or `mp4`. |
| `--clip-fps` | input FPS | Clip mode only. Output video FPS. |
| `--video-codec` | auto | Clip mode only. Override video codec. |
| `--audio-codec` | auto | Clip mode only. Override audio codec. |
| `--video-crf` | encoder default | Clip mode only. Video CRF/quality value passed to the encoder. |
| `--video-bitrate` | encoder default | Clip mode only. Video bitrate, e.g. `2500k` or `4m`. |
| `--audio-bitrate` | encoder default | Clip mode only. Audio bitrate, e.g. `128k`. |
| `--preset` | encoder default | Clip mode only. Encoder preset when supported. |
| `-w, --workers` | `min(4, #videos)` | Videos processed in parallel. |
| `-r, --recursive` | off | Recurse into subdirectories. |
| `--ext` | `mp4,webm` | Extensions to scan for. |
| `--flat` | off | Write all outputs flat, prefixed by video name, instead of one subdir per video. |
| `--dry-run` | off | List videos/clips/estimated counts, then exit. |
| `-v, --verbose` | off | Verbose logging. |

### Output Layout

By default each video gets its own subdirectory. The subdirectory name is derived
from the video stem, with Windows-unsafe characters and trailing spaces/dots
sanitized.

Sample mode filenames are the frame timestamp (`HH-MM-SS.mmm.<ext>`, `.png` or
`.jpg`), so they sort chronologically:

```text
out/
  my_long_video/
    00-01-00.123.png
    00-01-05.064.png
    ...
```

Clip mode filenames include the clip index and requested range:

```text
out/
  my_long_video/
    clip_000_00-01-00.000_00-05-20.000.webm
```

With `--flat`, files go directly in the output dir and are prefixed by the
sanitized video stem.

## Layout

```text
tools/video_sampler/
  main.py                     CLI entry point and orchestration
  components/
    clip_exporter.py          PyAV clip export + container conversion engine
    clip_parser.py            parse + validate clip ranges
    video_discovery.py        find video files in a dir / single file
    frame_sampler.py          PyAV decode + uniform per-minute frame sampling
  lib/
    timecode.py               timecode parse/format helpers
    logging_utils.py          logging setup
  requirements.txt
```
