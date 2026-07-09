#!/usr/bin/env python3
"""Auto-crop face + upper body from images, for LoRA training data.

Pipeline per image: YOLO detects the face bbox (largest face wins), SAM
segments the head incl. hair (box prompt) and the whole person (point
prompt, with an 8-direction probe fallback), then the crop window is
chosen by scoring candidate sizes and positions with

    score = person_pixels_inside - bg_penalty * background_pixels_inside

so the window grows only while the area it adds is mostly body (not
white space), and it slides toward the body mass in whatever direction
it lies (standing, lying, foreshortened). Hard constraints: the whole
head (hair included) stays inside, the face center stays in the central
band of the window, and face height stays between --face-frac-min and
--face-frac-max of crop height.

Crops are cut from the original pixels at the requested aspect ratio and
never resized, so pixel size is maximized and output dimensions vary per
image. Images with no detectable face are skipped with a warning; each
result logs the crop size, achieved face-height fraction, and body fill.

Models (same as the vn_face_reference workflows; resolved relative to
this repo, no downloads):
    models/ultralytics/bbox/face_yolov8m.pt   YOLO face bbox
    models/sams/sam_vit_b_01ec64.pth          SAM ViT-B masks

Tuning:
    --face-frac-min  floor on face size = ceiling on crop size. Full-body
                     sources usually land exactly here (biggest crop).
    --face-frac-max  cap on face size, for close-up sources.
    --bg-penalty     size vs white-space trade-off: the crop grows only
                     while added area is > penalty/(1+penalty) body
                     pixels. 0 = always take the biggest crop; raise it
                     for tighter, body-hugging crops.

Examples:
    # square crops, defaults
    python scripts/auto_crop_face_body.py ./raw ./crops
    # portrait 2:3, allow bigger crops, tolerate less background
    python scripts/auto_crop_face_body.py ./raw ./crops \\
        --ratio 2:3 --face-frac-min 0.25 --bg-penalty 0.6
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parent.parent
FACE_MODEL = REPO / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"
SAM_MODEL = REPO / "models" / "sams" / "sam_vit_b_01ec64.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JPEG_QUALITY = 95
N_SIZES = 8          # candidate crop sizes between the two face-frac bounds
GRID_STEPS = 40      # placement search resolution per axis
FACE_MARGIN = 0.03   # min gap between face bbox and crop edge, in crop heights
TIE_BREAK = 0.5      # weak pull toward face-centered-x / just-above-head-y


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_face(yolo, rgb: np.ndarray, conf: float):
    """Largest face bbox as (x1, y1, x2, y2) floats, or None."""
    res = yolo.predict(source=rgb[..., ::-1], conf=conf, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(areas.argmax())]


def sam_masks(predictor, rgb: np.ndarray, face):
    """Return (head_mask, person_mask) from one SAM embedding."""
    h, w = rgb.shape[:2]
    fx1, fy1, fx2, fy2 = face
    fw, fh = fx2 - fx1, fy2 - fy1
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    predictor.set_image(rgb)

    # Head incl. hair: box prompt expanded around the face bbox.
    head_box = np.array([
        max(0, fx1 - 0.4 * fw), max(0, fy1 - 0.8 * fh),
        min(w, fx2 + 0.4 * fw), min(h, fy2 + 0.3 * fh),
    ])
    masks, scores, _ = predictor.predict(box=head_box, multimask_output=True)
    head = masks[int(scores.argmax())]

    # Person: face-center point, take SAM's largest (whole-object) mask.
    pt = np.array([[fcx, fcy]], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=pt, point_labels=np.ones(1, dtype=np.int64),
        multimask_output=True)
    areas = masks.reshape(3, -1).sum(axis=1)
    ok = scores > 0.5
    person = masks[int(np.where(ok, areas, -1).argmax()) if ok.any()
                   else int(scores.argmax())]

    # Fallback: mask came back head-sized -> probe 8 directions for the body.
    if person.sum() < 3 * head.sum():
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            px, py = fcx + 2.2 * fh * dx, fcy + 2.2 * fh * dy
            if not (0 <= px < w and 0 <= py < h):
                continue
            pts = np.array([[fcx, fcy], [px, py]], dtype=np.float32)
            m, s, _ = predictor.predict(
                point_coords=pts, point_labels=np.ones(2, dtype=np.int64),
                multimask_output=True)
            cand = m[int(s.argmax())]
            if cand[int(py), int(px)] and (cand & head).any() \
                    and cand.sum() < 0.9 * h * w:
                person = person | cand
    return head, person | head


def compute_crop(shape, face, head, person, frac_min, frac_max,
                 ratio, bg_penalty):
    """Return (x0, y0, cw, ch, coverage) for the best-scoring window."""
    h, w = shape[:2]
    fx1, fy1, fx2, fy2 = [float(v) for v in face]
    fh = fy2 - fy1
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    rows = np.flatnonzero(head.any(axis=1))
    cols = np.flatnonzero(head.any(axis=0))
    if rows.size:
        hy1, hy2 = int(rows[0]), int(rows[-1])
        hx1, hx2 = int(cols[0]), int(cols[-1])
    else:
        hx1, hy1, hx2, hy2 = fx1, fy1, fx2, fy2
    head_top = hy1

    ch_hi = min(fh / frac_min, h, w / ratio)          # face at lower bound
    ch_lo = min(max(fh / frac_max, 1.2 * fh), ch_hi)  # face at cap

    sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat[1:, 1:] = person.cumsum(0).cumsum(1)

    def count(x0, y0, cw, ch):
        return sat[y0 + ch, x0 + cw] - sat[y0, x0 + cw] \
             - sat[y0 + ch, x0] + sat[y0, x0]

    best = None
    n = N_SIZES if ch_hi - ch_lo > 2 else 1
    for ch_f in np.linspace(ch_hi, ch_lo, n):  # biggest first wins ties
        ch, cw = int(ch_f), int(ch_f * ratio)
        if cw > w or ch > h or cw < 2 or ch < 2:
            continue
        m = FACE_MARGIN * ch
        m2 = 0.02 * ch  # margin around the head mask (hair included)
        # face fully inside with margin AND face center in the central band
        x_lo = int(max(0, fx2 + m - cw, fcx - 0.80 * cw))
        x_hi = int(min(w - cw, fx1 - m, fcx - 0.20 * cw))
        y_lo = int(max(0, fy2 + m - ch, fcy - 0.85 * ch))
        y_hi = int(min(h - ch, fy1 - m, fcy - 0.15 * ch))
        # whole head (hair) inside too, unless that leaves no feasible spot
        hx_lo, hx_hi = int(max(x_lo, hx2 + m2 - cw)), int(min(x_hi, hx1 - m2))
        hy_lo, hy_hi = int(max(y_lo, hy2 + m2 - ch)), int(min(y_hi, hy1 - m2))
        if hx_hi >= hx_lo:
            x_lo, x_hi = hx_lo, hx_hi
        if hy_hi >= hy_lo:
            y_lo, y_hi = hy_lo, hy_hi
        if x_hi < x_lo:
            x_lo = x_hi = int(np.clip(fcx - cw / 2, 0, w - cw))
        if y_hi < y_lo:
            y_lo = y_hi = int(np.clip(fcy - ch / 2, 0, h - ch))
        ix = fcx - cw / 2                     # tie-break ideals: centered x,
        iy = head_top - 0.06 * ch             # crop top just above the hair
        sx = max(1, (x_hi - x_lo) // GRID_STEPS)
        sy = max(1, (y_hi - y_lo) // GRID_STEPS)
        for y0 in range(y_lo, y_hi + 1, sy):
            for x0 in range(x_lo, x_hi + 1, sx):
                p = count(x0, y0, cw, ch)
                score = (1 + bg_penalty) * p - bg_penalty * cw * ch \
                    - TIE_BREAK * (abs(x0 - ix) + abs(y0 - iy))
                if best is None or score > best[0]:
                    best = (score, x0, y0, cw, ch, p)
    _, x0, y0, cw, ch, p = best
    return x0, y0, cw, ch, p / (cw * ch)


def parse_ratio(text: str) -> float:
    rw, rh = text.split(":")
    return float(rw) / float(rh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="an image file, or a directory of images "
                             "(jpg/png/webp/bmp/tiff)")
    parser.add_argument("output_dir", type=Path,
                        help="directory for <stem>_crop.jpg outputs "
                             "(created if missing)")
    parser.add_argument("--face-frac-min", type=float, default=0.30,
                        help="lower bound on face height / crop height; "
                             "sets the biggest allowed crop (default 0.30)")
    parser.add_argument("--face-frac-max", type=float, default=0.50,
                        help="upper bound on face height / crop height; "
                             "sets the smallest crop considered (default 0.50)")
    parser.add_argument("--ratio", type=parse_ratio, default="1:1",
                        help="crop aspect ratio as W:H, e.g. 1:1, 2:3, 3:4 "
                             "(default 1:1)")
    parser.add_argument("--bg-penalty", type=float, default=0.35,
                        help="cost of each background pixel inside the crop; "
                             "0 always takes the biggest crop, higher values "
                             "give tighter body-hugging crops (default 0.35)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO face-detection confidence threshold; lower "
                             "finds harder faces at the risk of false boxes "
                             "(default 0.3)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="inference device; auto picks cuda > mps > cpu")
    args = parser.parse_args()

    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    else:
        files = [args.input]
    if not files:
        parser.error(f"no images found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor

    device = pick_device(args.device)
    yolo = YOLO(str(FACE_MODEL))
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_MODEL)).to(device)
    predictor = SamPredictor(sam)
    print(f"device={device}  ratio={args.ratio:.4f}  "
          f"face_frac=[{args.face_frac_min}, {args.face_frac_max}]  "
          f"bg_penalty={args.bg_penalty}")

    done = skipped = 0
    for path in files:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        rgb = np.asarray(img)
        face = detect_face(yolo, rgb, args.conf)
        if face is None:
            print(f"SKIP {path.name}: no face detected")
            skipped += 1
            continue
        head, person = sam_masks(predictor, rgb, face)
        x0, y0, cw, ch, cov = compute_crop(
            rgb.shape, face, head, person,
            args.face_frac_min, args.face_frac_max,
            args.ratio, args.bg_penalty)
        crop = img.crop((x0, y0, x0 + cw, y0 + ch))
        out = args.output_dir / f"{path.stem}_crop.jpg"
        crop.save(out, quality=JPEG_QUALITY)
        face_pct = (face[3] - face[1]) / ch
        print(f"OK   {path.name}: crop {cw}x{ch} at ({x0},{y0}), "
              f"face {face_pct:.0%} of height, body fill {cov:.0%}")
        done += 1
    print(f"done: {done} cropped, {skipped} skipped -> {args.output_dir}")


if __name__ == "__main__":
    main()
