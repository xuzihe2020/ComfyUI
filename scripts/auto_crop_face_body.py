#!/usr/bin/env python3
r"""Auto-crop face + upper body from images, for LoRA training data.

Basic usage:
    python scripts/auto_crop_face_body.py INPUT OUTPUT_DIR [options]

INPUT can be one image file or a directory. If it is a directory, all
files with these extensions are processed:

    .jpg .jpeg .png .webp .bmp .tif .tiff

OUTPUT_DIR is created if it does not exist. Each successful crop is saved
as a JPEG named:

    <source-stem>_crop.jpg

No crop is resized. The script cuts the largest useful crop from the
original pixels at the requested aspect ratio, so output dimensions vary
from image to image. Images with no detectable face are skipped with a
warning.

Pipeline per image:
    1. YOLO detects face boxes and the largest face wins.
    2. YOLO person segmentation finds the body/person mask containing
       that face.
    3. The head/hair safety mask is derived from person-mask pixels near
       the face, with a deterministic expanded-face fallback only if the
       person mask is empty there.
    4. Candidate crop sizes and positions are chosen by hard safety first:
       include the full head/hair mask, then choose the largest possible
       crop, and then prefer the placement with the most body/person
       pixels inside it.

Hard crop constraints:
    - the whole detected head/hair mask should stay inside the crop
    - the face box should stay inside with a small margin
    - the face center should stay in a central band of the crop
    - face height should be between --face-frac-min and --face-frac-max
      of the final crop height

Required local models, resolved through extra_model_paths.yaml when it
exists, then through repo-local fallbacks:
    ultralytics_bbox/face_yolov8m.pt
        YOLO face detector. With the default external model config this is:
        C:\Users\Tony Xu\workspace\comfyui_models\ultralytics\bbox\face_yolov8m.pt

    ultralytics_segm/person_yolov8m-seg.pt
        YOLO person segmentation model. With the default external model config
        this is:
        C:\Users\Tony Xu\workspace\comfyui_models\ultralytics\segm\person_yolov8m-seg.pt

Common PowerShell examples:
    # Default square crops from every image in a folder.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops

    # Process one image only.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw\person01.jpg .\crops

    # Portrait 2:3 crops for LoRA training.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --ratio 2:3

    # Larger crops that include more torso/body.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --face-frac-min 0.25

    # Tighter crops for images with too much whitespace.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --bg-penalty 0.6

    # Close-up friendly crops where the face may occupy more of the image.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --face-frac-max 0.60

    # Try harder to detect small or difficult faces.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --conf 0.15

    # Force a device. By default, --device auto chooses cuda > mps > cpu.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --device cuda

    # Write debug masks to the default directory:
    #   <output_dir>\debug_masks
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --debug

    # Write debug masks to a custom directory.
    .\.venv\Scripts\python.exe scripts\auto_crop_face_body.py `
        .\raw .\crops --debug-dir .\mask_debug

Argument guide:
    --ratio W:H
        Crop aspect ratio, such as 1:1, 2:3, 3:4, or 4:5.
        Default: 1:1.

    --face-frac-min FLOAT
        Lower bound for face height divided by crop height. This sets the
        biggest crop the script may choose. Lower values allow larger
        body crops. Full-body or upper-body sources often land at this
        bound because the script tries to keep useful body pixels.
        Default: 0.30.

    --face-frac-max FLOAT
        Upper bound for face height divided by crop height. This sets the
        smallest crop the script considers. Higher values allow tighter
        close-ups where the face occupies more of the crop.
        Default: 0.50.

    --bg-penalty FLOAT
        Legacy placement scoring knob. The crop now prioritizes largest
        head-safe crop first, so this is only used as a same-size
        tie-breaker when comparing body fill against background fill.
        Default: 0.35.

    --conf FLOAT
        YOLO face detection confidence threshold. Lower this if faces are
        being missed; raise it if false face boxes are being selected.
        Default: 0.30.

    --device {auto,cpu,cuda,mps}
        Inference device. "auto" chooses cuda, then mps, then cpu.
        Default: auto.

    --debug
        Save mask PNGs for each processed image. The default debug
        directory is:

            <output_dir>/debug_masks

        Files written:
            <source-stem>_face_mask.png
            <source-stem>_head_mask.png
            <source-stem>_body_mask.png

        The face mask is the YOLO face bounding box. The body mask is
        YOLO person segmentation. The head mask is the connected person
        segment near the detected face.

    --debug-dir PATH
        Save debug masks to PATH. Passing --debug-dir also enables debug
        mask output, even if --debug is omitted.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
EXTRA_MODEL_PATHS = REPO / "extra_model_paths.yaml"
FACE_MODEL_NAME = "face_yolov8m.pt"
PERSON_SEG_MODEL_NAME = "person_yolov8m-seg.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
JPEG_QUALITY = 95
N_SIZES = 8          # candidate crop sizes between the two face-frac bounds
GRID_STEPS = 40      # placement search resolution per axis
FACE_MARGIN = 0.03   # min gap between face bbox and crop edge, in crop heights
HEAD_SIDE_MARGIN = 0.03
HEAD_TOP_MARGIN = 0.06
HEAD_BOTTOM_MARGIN = 0.02
HEAD_BOX_SIDE_EXPAND = 0.55
HEAD_BOX_TOP_EXPAND = 0.75
HEAD_BOX_BOTTOM_EXPAND = 0.55
HEAD_COVERAGE_MIN = 0.995
DEBUG_DIR_NAME = "debug_masks"


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_config_path(path_text: str, yaml_dir: Path, base_path: Path | None):
    path = Path(os.path.expandvars(os.path.expanduser(path_text)))
    if base_path is not None:
        path = base_path / path
    elif not path.is_absolute():
        path = yaml_dir / path
    return path.resolve()


def extra_model_dirs(folder_names: set[str]) -> list[Path]:
    """Return model folders from ComfyUI's extra_model_paths.yaml."""
    if not EXTRA_MODEL_PATHS.is_file():
        return []

    with EXTRA_MODEL_PATHS.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    yaml_dir = EXTRA_MODEL_PATHS.parent
    found = []
    for section in config.values():
        if not section:
            continue
        base_path = section.get("base_path")
        if base_path:
            base_path = resolve_config_path(base_path, yaml_dir, None)
        is_default = bool(section.get("is_default", False))
        for key, value in section.items():
            if key in ("base_path", "is_default") or key not in folder_names:
                continue
            for item in str(value).splitlines():
                if not item:
                    continue
                folder = resolve_config_path(item, yaml_dir, base_path)
                if is_default:
                    found.insert(0, folder)
                else:
                    found.append(folder)
    return found


def resolve_model_path(model_name: str, folder_names: set[str],
                       fallback_paths: list[Path]) -> Path:
    candidates = [folder / model_name for folder in extra_model_dirs(folder_names)]
    candidates.extend(fallback_paths)
    for path in candidates:
        if path.is_file():
            return path

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"could not find {model_name}. Searched:\n  {searched}")


def resolve_models() -> tuple[Path, Path]:
    face_model = resolve_model_path(
        FACE_MODEL_NAME,
        {"ultralytics_bbox"},
        [REPO / "models" / "ultralytics" / "bbox" / FACE_MODEL_NAME],
    )
    person_seg_model = resolve_model_path(
        PERSON_SEG_MODEL_NAME,
        {"ultralytics_segm"},
        [REPO / "models" / "ultralytics" / "segm" / PERSON_SEG_MODEL_NAME],
    )
    return face_model, person_seg_model


def detect_face(yolo, rgb: np.ndarray, conf: float):
    """Largest face bbox as (x1, y1, x2, y2) floats, or None."""
    res = yolo.predict(source=rgb[..., ::-1], conf=conf, verbose=False)[0]
    if len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(areas.argmax())]


def mask_bbox(mask: np.ndarray):
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def is_page_sized_mask(mask: np.ndarray) -> bool:
    """Reject segmentation masks that are effectively the whole image/page."""
    h, w = mask.shape[:2]
    box = mask_bbox(mask)
    if box is None:
        return True
    x1, y1, x2, y2 = box
    area_frac = mask.sum() / (h * w)
    bbox_frac = ((x2 - x1 + 1) * (y2 - y1 + 1)) / (h * w)
    touches = sum((x1 <= 0.02 * w, y1 <= 0.02 * h,
                   x2 >= 0.98 * w, y2 >= 0.98 * h))
    return area_frac > 0.60 or (bbox_frac > 0.90 and touches >= 3)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(img) > 0


def detect_person_mask(yolo, rgb: np.ndarray, face, conf: float) -> np.ndarray:
    """Return the YOLO person segmentation mask containing the detected face."""
    h, w = rgb.shape[:2]
    fx1, fy1, fx2, fy2 = face
    fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    result = yolo.predict(source=rgb[..., ::-1], conf=conf, verbose=False)[0]
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((h, w), dtype=bool)

    masks = result.masks.data.cpu().numpy() > 0.5
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int) \
        if result.boxes.cls is not None else np.full(len(masks), -1)
    names = result.names or {}

    face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    candidates = []
    for i, raw_mask in enumerate(masks):
        cls_name = str(names.get(int(classes[i]), "")).lower()
        mask = resize_mask(raw_mask, w, h)
        x1, y1, x2, y2 = boxes[i]
        bx1, by1 = max(x1, fx1), max(y1, fy1)
        bx2, by2 = min(x2, fx2), min(y2, fy2)
        box_overlap = max(0, bx2 - bx1) * max(0, by2 - by1) / face_area
        mask_overlap = mask[int(max(0, fy1)):int(min(h, fy2)),
                            int(max(0, fx1)):int(min(w, fx2))].sum() / face_area
        center_hit = bool(mask[int(np.clip(fcy, 0, h - 1)),
                               int(np.clip(fcx, 0, w - 1))])
        is_person = cls_name in ("", "person") or len(names) == 1
        area = int(mask.sum())
        candidates.append((is_person, center_hit, mask_overlap, box_overlap,
                           area, mask))

    candidates = [c for c in candidates if c[0]] or candidates
    return max(candidates, key=lambda c: c[:-1])[-1]


def fallback_head_mask(shape, face) -> np.ndarray:
    h, w = shape[:2]
    fx1, fy1, fx2, fy2 = [float(v) for v in face]
    fw, fh = fx2 - fx1, fy2 - fy1
    x1 = int(max(0, fx1 - HEAD_BOX_SIDE_EXPAND * fw))
    y1 = int(max(0, fy1 - HEAD_BOX_TOP_EXPAND * fh))
    x2 = int(min(w, fx2 + HEAD_BOX_SIDE_EXPAND * fw))
    y2 = int(min(h, fy2 + HEAD_BOX_BOTTOM_EXPAND * fh))
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def head_mask_from_person(person: np.ndarray, face) -> np.ndarray:
    """Use person segmentation near the face as the head/hair safety mask."""
    h, w = person.shape[:2]
    fx1, fy1, fx2, fy2 = [float(v) for v in face]
    fw, fh = fx2 - fx1, fy2 - fy1
    x1 = int(max(0, fx1 - HEAD_BOX_SIDE_EXPAND * fw))
    y1 = int(max(0, fy1 - HEAD_BOX_TOP_EXPAND * fh))
    x2 = int(min(w, fx2 + HEAD_BOX_SIDE_EXPAND * fw))
    y2 = int(min(h, fy2 + HEAD_BOX_BOTTOM_EXPAND * fh))
    roi = np.zeros_like(person, dtype=bool)
    roi[y1:y2, x1:x2] = True
    head = person & roi
    if not head.any():
        return fallback_head_mask(person.shape, face)

    labels, n = ndimage.label(head)
    if n == 0:
        return fallback_head_mask(person.shape, face)

    face_roi = np.zeros_like(person, dtype=bool)
    face_roi[int(max(0, fy1)):int(min(h, fy2)),
             int(max(0, fx1)):int(min(w, fx2))] = True
    keep = []
    for label in range(1, n + 1):
        comp = labels == label
        if (comp & face_roi).any():
            keep.append(label)
    if not keep:
        px, py = int((fx1 + fx2) / 2), int((fy1 + fy2) / 2)
        if 0 <= px < w and 0 <= py < h and labels[py, px] > 0:
            keep = [int(labels[py, px])]
    if not keep:
        keep = [max(range(1, n + 1), key=lambda label: (labels == label).sum())]

    return np.isin(labels, keep)


def save_debug_masks(path: Path, debug_dir: Path, shape, face, head, person):
    """Write mask PNGs for the YOLO face bbox and crop masks."""
    h, w = shape[:2]
    fx1, fy1, fx2, fy2 = [int(round(v)) for v in face]
    x1, x2 = sorted((int(np.clip(fx1, 0, w)), int(np.clip(fx2, 0, w))))
    y1, y2 = sorted((int(np.clip(fy1, 0, h)), int(np.clip(fy2, 0, h))))

    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[y1:y2, x1:x2] = 255

    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    Image.fromarray(face_mask).save(debug_dir / f"{stem}_face_mask.png")
    Image.fromarray(head.astype(np.uint8) * 255).save(
        debug_dir / f"{stem}_head_mask.png")
    Image.fromarray(person.astype(np.uint8) * 255).save(
        debug_dir / f"{stem}_body_mask.png")


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
    ch_hi = min(fh / frac_min, h, w / ratio)          # face at lower bound
    ch_lo = min(max(fh / frac_max, 1.2 * fh), ch_hi)  # face at cap

    sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat[1:, 1:] = person.cumsum(0).cumsum(1)
    head_sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    head_sat[1:, 1:] = head.cumsum(0).cumsum(1)
    head_total = max(1, int(head.sum()))

    def count(table, x0, y0, cw, ch):
        return table[y0 + ch, x0 + cw] - table[y0, x0 + cw] \
             - table[y0 + ch, x0] + table[y0, x0]

    def intersect_ranges(*ranges):
        lo = max(r[0] for r in ranges)
        hi = min(r[1] for r in ranges)
        if hi < lo:
            return None
        return lo, hi

    def required_range(min_v, max_v, size):
        return max_v - size, min_v

    best = fallback = None
    n = N_SIZES if ch_hi - ch_lo > 2 else 1
    for ch_f in np.linspace(ch_hi, ch_lo, n):  # biggest first wins ties
        ch, cw = int(ch_f), int(ch_f * ratio)
        if cw > w or ch > h or cw < 2 or ch < 2:
            continue
        hm_x = HEAD_SIDE_MARGIN * ch
        hm_top = HEAD_TOP_MARGIN * ch
        hm_bottom = HEAD_BOTTOM_MARGIN * ch

        image_x = (0, w - cw)
        image_y = (0, h - ch)
        head_margin_x = required_range(
            max(0, hx1 - hm_x), min(w, hx2 + 1 + hm_x), cw)
        head_margin_y = required_range(
            max(0, hy1 - hm_top), min(h, hy2 + 1 + hm_bottom), ch)
        head_x = required_range(hx1, hx2 + 1, cw)
        head_y = required_range(hy1, hy2 + 1, ch)
        center_x = (fcx - 0.80 * cw, fcx - 0.20 * cw)
        x_rng = intersect_ranges(image_x, head_margin_x)
        y_rng = intersect_ranges(image_y, head_margin_y)
        margin_ok = x_rng is not None and y_rng is not None
        if not margin_ok:
            x_rng = intersect_ranges(image_x, head_x)
            y_rng = intersect_ranges(image_y, head_y)
        if x_rng is None or y_rng is None:
            x_rng, y_rng = image_x, image_y

        centered_x = intersect_ranges(x_rng, center_x)
        if centered_x is not None:
            x_rng = centered_x

        x_lo, x_hi = int(np.ceil(x_rng[0])), int(np.floor(x_rng[1]))
        y_lo, y_hi = int(np.ceil(y_rng[0])), int(np.floor(y_rng[1]))
        if x_hi < x_lo:
            x_lo = x_hi = int(np.clip(round((x_rng[0] + x_rng[1]) / 2),
                                      0, w - cw))
        if y_hi < y_lo:
            y_lo = y_hi = int(np.clip(round((y_rng[0] + y_rng[1]) / 2),
                                      0, h - ch))
        ix = fcx - cw / 2
        sx = max(1, (x_hi - x_lo) // GRID_STEPS)
        sy = max(1, (y_hi - y_lo) // GRID_STEPS)
        xs = list(range(x_lo, x_hi + 1, sx))
        ys = list(range(y_lo, y_hi + 1, sy))
        if xs[-1] != x_hi:
            xs.append(x_hi)
        if ys[-1] != y_hi:
            ys.append(y_hi)
        for y0 in ys:
            for x0 in xs:
                p = count(sat, x0, y0, cw, ch)
                head_inside = count(head_sat, x0, y0, cw, ch)
                head_cov = head_inside / head_total
                body_score = (1 + bg_penalty) * p - bg_penalty * cw * ch
                fallback_key = (
                    head_cov, margin_ok, cw * ch, body_score, y0,
                    -abs(x0 - ix))
                if fallback is None or fallback_key > fallback[0]:
                    fallback = (fallback_key, x0, y0, cw, ch, p)
                if head_cov < HEAD_COVERAGE_MIN:
                    continue
                key = (cw * ch, margin_ok, body_score, y0, -abs(x0 - ix))
                if best is None or key > best[0]:
                    best = (key, x0, y0, cw, ch, p)
    if best is None:
        best = fallback
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
    parser.add_argument("--debug", action="store_true",
                        help="write debug PNG masks for each processed image")
    parser.add_argument("--debug-dir", type=Path,
                        help="directory for debug masks; default is "
                             "<output_dir>/debug_masks")
    args = parser.parse_args()

    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    else:
        files = [args.input]
    if not files:
        parser.error(f"no images found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_enabled = args.debug or args.debug_dir is not None
    if debug_enabled:
        debug_dir = args.debug_dir or args.output_dir / DEBUG_DIR_NAME
    else:
        debug_dir = None

    from ultralytics import YOLO

    device = pick_device(args.device)
    face_model, person_seg_model = resolve_models()
    yolo = YOLO(str(face_model))
    person_yolo = YOLO(str(person_seg_model))
    print(f"device={device}  ratio={args.ratio:.4f}  "
          f"face_frac=[{args.face_frac_min}, {args.face_frac_max}]  "
          f"bg_penalty={args.bg_penalty}")
    print(f"face_model={face_model}")
    print(f"person_seg_model={person_seg_model}")
    if debug_dir is not None:
        print(f"debug_masks={debug_dir}")

    done = skipped = 0
    for path in files:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        rgb = np.asarray(img)
        face = detect_face(yolo, rgb, args.conf)
        if face is None:
            print(f"SKIP {path.name}: no face detected")
            skipped += 1
            continue
        person = detect_person_mask(person_yolo, rgb, face, args.conf)
        head = head_mask_from_person(person, face)
        person = person | head
        if debug_dir is not None:
            save_debug_masks(path, debug_dir, rgb.shape, face, head, person)
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
