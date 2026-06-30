#!/usr/bin/env python3
"""auto_image_cropper - batch resize (aspect-preserving) or center-crop images.

Examples
--------
Resize every image in a folder to 80% of its size::

    python tools/auto_image_cropper/main.py resize ./in -o ./out --scale 0.8

Resize to 800px wide, height derived to keep the aspect ratio::

    python tools/auto_image_cropper/main.py resize ./in -o ./out --width 800

Center-crop to an exact size, or by symmetric margins (equivalent here)::

    python tools/auto_image_cropper/main.py crop ./in -o ./out --size 1000 1500
    python tools/auto_image_cropper/main.py crop ./in -o ./out --margin 12 18
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Make `components` and `lib` importable regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from components.cropper import crop_image  # noqa: E402
from components.image_discovery import DEFAULT_EXTENSIONS, discover_images  # noqa: E402
from components.resizer import resize_image, validate_resize_spec  # noqa: E402
from lib.errors import ImageOpError  # noqa: E402
from lib.imaging import load_image, save_image  # noqa: E402
from lib.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger("auto_image_cropper")


@dataclass
class Result:
    path: Path
    ok: bool
    src: tuple | None = None
    dst: tuple | None = None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_image_cropper",
        description="Batch resize (aspect-preserving) or center-crop images.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input", help="Image file or directory of images.")
        sp.add_argument("-o", "--output", required=True, help="Output directory.")
        sp.add_argument(
            "-r",
            "--recursive",
            action="store_true",
            help="Recurse into subdirectories (relative paths are preserved).",
        )
        sp.add_argument(
            "--ext",
            default=None,
            help="Comma-separated extensions to scan for "
            "(default: png,jpg,jpeg,webp,bmp,tif,tiff).",
        )
        sp.add_argument(
            "-q",
            "--quality",
            type=int,
            default=95,
            help="Quality for lossy outputs (JPEG/WebP), 0-100. Ignored otherwise.",
        )
        sp.add_argument(
            "-w",
            "--workers",
            type=int,
            default=None,
            help="Images processed in parallel (default: min(8, #cpus)).",
        )
        sp.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")

    pr = sub.add_parser(
        "resize",
        help="Aspect-preserving resize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(pr)
    pr.add_argument("--scale", type=float, help="Uniform scale factor, e.g. 0.8 = 80%%.")
    pr.add_argument("--width", type=int, help="Target width in pixels.")
    pr.add_argument("--height", type=int, help="Target height in pixels.")

    pc = sub.add_parser(
        "crop",
        help="Center crop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common(pc)
    grp = pc.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Target crop size, taken from the center.",
    )
    grp.add_argument(
        "--margin",
        type=int,
        nargs=2,
        metavar=("MW", "MH"),
        help="Symmetric margins: trim MW off left+right, MH off top+bottom.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    if not (0 <= args.quality <= 100):
        logger.error("--quality must be in 0..100 (got %s)", args.quality)
        return 2

    if args.command == "resize":
        try:
            validate_resize_spec(args.scale, args.width, args.height)
        except ImageOpError as exc:
            logger.error("%s", exc)
            return 2
        scale, width, height = args.scale, args.width, args.height

        def operate(img):
            return resize_image(img, scale=scale, width=width, height=height)

        op_desc = (
            f"resize scale={scale}"
            if scale is not None
            else f"resize to width={width} height={height}"
        )
    else:  # crop
        size = tuple(args.size) if args.size else None
        margin = tuple(args.margin) if args.margin else None

        def operate(img):
            return crop_image(img, size=size, margin=margin)

        op_desc = f"crop size={size}" if size else f"crop margin={margin}"

    extensions = (
        tuple(e.strip() for e in args.ext.split(",") if e.strip())
        if args.ext
        else DEFAULT_EXTENSIONS
    )
    try:
        images = discover_images(args.input, recursive=args.recursive, extensions=extensions)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    if not images:
        logger.error("no images found under %s", args.input)
        return 2

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    if input_path.is_file():
        items = [(input_path, Path(input_path.name))]
    else:
        items = [(p, p.relative_to(input_path)) for p in images]

    logger.info("%s: %d image(s) -> %s", op_desc, len(items), out_dir)

    workers = args.workers or min(8, (os.cpu_count() or 1))
    workers = max(1, min(workers, len(items)))

    def process(image_path: Path, rel: Path) -> Result:
        try:
            img = load_image(image_path)
            src = (img.width, img.height)
            result_img = operate(img)
            save_image(result_img, out_dir / rel, quality=args.quality)
            return Result(image_path, True, src, (result_img.width, result_img.height))
        except ImageOpError as exc:
            return Result(image_path, False, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return Result(image_path, False, error=f"{type(exc).__name__}: {exc}")

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, p, rel): p for p, rel in items}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r.ok:
                logger.debug(
                    "%s  %dx%d -> %dx%d", r.path.name, r.src[0], r.src[1], r.dst[0], r.dst[1]
                )
            else:
                logger.warning("%s: %s", r.path.name, r.error)

    return _report(results, out_dir)


def _report(results: list[Result], out_dir: Path) -> int:
    results.sort(key=lambda r: str(r.path))
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print("\n=== Summary ===")
    print(f"Processed {len(ok)}/{len(results)} image(s) into {out_dir}")
    if ok:
        s = ok[0]
        print(f"  e.g. {s.path.name}: {s.src[0]}x{s.src[1]} -> {s.dst[0]}x{s.dst[1]}")
    for r in failed:
        print(f"  FAILED {r.path.name}: {r.error}")
    if failed:
        print(f"{len(failed)} image(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
