# auto_image_cropper

Batch **resize** (aspect-ratio preserving) or **center-crop** images, built on
Pillow (already a ComfyUI dependency). Works on a single image or a whole
directory; processes images in parallel.

Two subcommands: `resize` and `crop`.

## resize — never stretches, deforms, or crops

Pick exactly one way to specify the target:

| Input | Behavior |
| --- | --- |
| `--scale 0.8` | Uniform 80% of each side. |
| `--width 800` | Scale to 800px wide; height derived to keep the aspect ratio. |
| `--height 600` | Scale to 600px tall; width derived to keep the aspect ratio. |
| `--width 800 --height 600` | Allowed **only if** `800×600` already matches the image's aspect ratio (a pure uniform scale). Otherwise it **raises** — honoring both would require stretching or cropping. |

```bash
python tools/auto_image_cropper/main.py resize ./in -o ./out --scale 0.8
python tools/auto_image_cropper/main.py resize ./in -o ./out --width 800
```

Resampling uses Lanczos (high quality for both down- and up-scaling).

## crop — center region only, two input modes

Both modes keep the **centered** region; they only differ in how you express it.
For a `1024×1536` image:

| Input | Result | What it trims |
| --- | --- | --- |
| `--size 1000 1500` | `1000×1500` | 12px off left & right, 18px off top & bottom |
| `--margin 12 18` | `1000×1500` | width-margin 12px off left & right, height-margin 18px off top & bottom |

```bash
python tools/auto_image_cropper/main.py crop ./in -o ./out --size 1000 1500
python tools/auto_image_cropper/main.py crop ./in -o ./out --margin 12 18
```

`--size` larger than the image, or `--margin` that removes the whole image, raises.

## Common arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `input` (positional) | — | Image file, or directory of images. |
| `-o, --output` | required | Output directory (relative paths preserved under it). |
| `-r, --recursive` | off | Recurse into subdirectories. |
| `--ext` | `png,jpg,jpeg,webp,bmp,tif,tiff` | Extensions to scan for. |
| `-q, --quality` | `95` | Quality for lossy outputs (JPEG/WebP), 0–100. Ignored for PNG. |
| `-w, --workers` | `min(8, #cpus)` | Images processed in parallel. |
| `-v, --verbose` | off | Verbose logging. |

Output keeps each input's format/extension. Errors are reported per image; the
batch continues and exits non-zero if any image failed.

## Layout

```
tools/auto_image_cropper/
  main.py                  CLI (subcommands) + batch orchestration
  components/
    image_discovery.py     find images in a dir / single file
    resizer.py             aspect-preserving resize + validation
    cropper.py             centered crop (size or margin) + validation
  lib/
    imaging.py             Pillow load/save, resampling filter
    errors.py              ImageOpError
    logging_utils.py       logging setup
  requirements.txt
```
