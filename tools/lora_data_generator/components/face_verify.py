"""Face identity scoring for generated LoRA images (InsightFace / ArcFace).

After each generation the runners score every saved image against the item's
primary character reference — the FIRST character-kind reference image — and
print one console line per image:

    <image name>: <score>

The score is cosine similarity between ArcFace embeddings (buffalo_l pack).
Rule of thumb: >0.5 same person, 0.3-0.5 borderline, <0.3 different person.
Calibrate on known-good pairs of your own character before trusting absolute
thresholds; scores degrade on heavily stylized/non-photoreal faces.

Scoring is best-effort by design: a missing insightface install, an unreadable
file, or a shot with no detectable face must never fail a paid generation run.
Those cases print a note instead of a score and are recorded in the debug log.

Model weights (~330 MB) download once to ~/.insightface/ by default; pass
--face-model-root (or set FACE_MODEL_ROOT in .env) to keep them in an external
models folder instead of the home cache.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from pathlib import Path
from typing import Any

_ANALYZER = None
_ANALYZER_ERROR: str | None = None
_REFERENCE_EMBEDDINGS: dict[Path, Any] = {}


def _get_analyzer(model_root: str | None = None):
    """Lazy singleton so generation runs pay the model load exactly once."""
    global _ANALYZER, _ANALYZER_ERROR
    if _ANALYZER is not None or _ANALYZER_ERROR is not None:
        return _ANALYZER
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        _ANALYZER_ERROR = (
            "insightface is not installed; skipping face similarity "
            "(install the repo requirements.txt into the venv)"
        )
        return None

    kwargs: dict[str, Any] = {
        "name": "buffalo_l",
        "allowed_modules": ["detection", "recognition"],
    }
    model_root = model_root or os.environ.get("FACE_MODEL_ROOT", "")
    if model_root:
        kwargs["root"] = model_root

    # Request only providers that exist; insightface defaults to CUDA and
    # warns loudly on machines without it (e.g. macOS).
    try:
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
    except ImportError:
        available = set()
    if "CUDAExecutionProvider" in available:
        kwargs["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        kwargs["providers"] = ["CPUExecutionProvider"]

    # skimage deprecation chatter from insightface's face alignment.
    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

    try:
        # insightface/onnxruntime chatter ("Applied providers", "find model")
        # would drown the `<image name>: <score>` lines; keep the console clean.
        with contextlib.redirect_stdout(io.StringIO()):
            analyzer = FaceAnalysis(**kwargs)
            analyzer.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as exc:  # noqa: BLE001 - init failure must not kill a run
        _ANALYZER_ERROR = f"face analyzer init failed: {exc}"
        return None
    _ANALYZER = analyzer
    return _ANALYZER


def _embedding(analyzer, image_path: Path):
    """Return (embedding, error). Uses the largest detected face in the image."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return None, "unreadable image"
    faces = analyzer.get(image)
    if not faces:
        return None, "no face detected"
    largest = max(
        faces,
        key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )
    return largest.normed_embedding, None


def _reference_embedding(analyzer, reference: Path):
    if reference not in _REFERENCE_EMBEDDINGS:
        _REFERENCE_EMBEDDINGS[reference] = _embedding(analyzer, reference)
    return _REFERENCE_EMBEDDINGS[reference]


def primary_character_reference(refs: list[tuple[str, Path]]) -> Path | None:
    """The first character-kind reference image; dressing refs never count."""
    for kind, path in refs:
        if kind == "character":
            return path
    return None


def similarity_score(
    image_a: Path, image_b: Path, model_root: str | None = None,
) -> tuple[float | None, str | None]:
    """Cosine similarity between the largest faces of two images."""
    analyzer = _get_analyzer(model_root)
    if analyzer is None:
        return None, _ANALYZER_ERROR

    import numpy as np

    embedding_a, error_a = _reference_embedding(analyzer, image_a)
    if embedding_a is None:
        return None, f"{image_a.name}: {error_a}"
    embedding_b, error_b = _embedding(analyzer, image_b)
    if embedding_b is None:
        return None, error_b
    return float(np.dot(embedding_a, embedding_b)), None


def score_against_reference(
    reference: Path | None,
    saved: list[Path],
    model_root: str | None = None,
) -> list[dict[str, Any]]:
    """Score saved images vs a reference; print `<name>: <score>` per image.

    Returns log entries. Never raises — a paid generation run must not fail
    because scoring did.
    """
    entries: list[dict[str, Any]] = []
    if not saved:
        return entries
    if reference is None:
        print("  face similarity: skipped (no character reference)")
        return entries
    if not reference.is_file():
        print(f"  face similarity: skipped (reference not found: {reference})")
        return entries

    for path in saved:
        try:
            score, note = similarity_score(reference, path, model_root)
        except Exception as exc:  # noqa: BLE001 - scoring must never fail the run
            score, note = None, f"face similarity error: {exc}"
        entry: dict[str, Any] = {
            "image": str(path),
            "reference": str(reference),
            "score": None if score is None else round(score, 4),
        }
        if note:
            entry["note"] = note
        entries.append(entry)
        print(f"  {path.name}: {note if score is None else f'{score:.3f}'}")
    return entries


def score_saved_images(
    refs: list[tuple[str, Path]],
    saved: list[Path],
    model_root: str | None = None,
) -> list[dict[str, Any]]:
    """Score saved images against the item's primary character reference."""
    return score_against_reference(primary_character_reference(refs), saved, model_root)


def verify_pair(image_a: Path, image_b: Path, model_root: str | None = None) -> int:
    """--mode verify: score two images and print the result. CLI exit code."""
    for path in (image_a, image_b):
        if not path.is_file():
            print(f"error: image not found: {path}")
            return 2
    score, note = similarity_score(image_a.resolve(), image_b.resolve(), model_root)
    if score is None:
        print(f"error: {note}")
        return 2
    if score > 0.5:
        verdict = "same person"
    elif score >= 0.3:
        verdict = "borderline"
    else:
        verdict = "different person"
    print(f"{image_a.name} vs {image_b.name}: {score:.3f}")
    print(f"verdict: {verdict} (>0.5 same, 0.3-0.5 borderline, <0.3 different)")
    return 0
