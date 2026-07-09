#!/usr/bin/env python3
"""Extract all embedded images from a PDF (or a folder of PDFs) as JPEGs.

Extraction reads the PDF's embedded image streams directly -- pages are
never rendered/rasterized -- so output quality is the maximum the file
can provide:

  - images stored as JPEG are copied byte-for-byte (zero re-encode)
  - all other encodings (JPEG 2000, PNG, TIFF, ...) are decoded once and
    saved as JPEG at quality 95

Output files are named <pdf-stem>-p<page>-<n>.jpg, where <page> is the
1-based page number and <n> is the image's index on that page. Images
repeated across pages (logos, watermarks) are extracted only once.

Tip: run `pdfimages -list file.pdf` (poppler) first to preview the
dimensions and encodings inside a PDF.

Requires the `pymupdf` package (already installed in this repo's .venv).

Examples:
    # one PDF
    python scripts/extract_pdf_images.py photos.pdf ./out
    # every *.pdf in a folder, into the same output dir
    python scripts/extract_pdf_images.py ~/Documents/scans ./out
"""

import argparse
from pathlib import Path

import pymupdf

JPEG_QUALITY = 95


def extract_pdf(pdf_path: Path, out_dir: Path) -> int:
    doc = pymupdf.open(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    for page in doc:
        for index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            out_path = out_dir / f"{pdf_path.stem}-p{page.number + 1:04d}-{index}.jpg"
            info = doc.extract_image(xref)
            if info["ext"] in ("jpg", "jpeg"):
                out_path.write_bytes(info["image"])
            else:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.alpha:
                    pix = pymupdf.Pixmap(pix, 0)
                if pix.colorspace is None or pix.colorspace.n not in (1, 3):
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                pix.save(out_path, jpg_quality=JPEG_QUALITY)
            count += 1
    doc.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="a PDF file, or a directory whose *.pdf files "
                             "are all processed")
    parser.add_argument("output_dir", type=Path,
                        help="directory to write JPEGs into (created if missing)")
    args = parser.parse_args()

    pdfs = sorted(args.input.glob("*.pdf")) if args.input.is_dir() else [args.input]
    if not pdfs:
        parser.error(f"no PDFs found in {args.input}")

    total = 0
    for pdf in pdfs:
        n = extract_pdf(pdf, args.output_dir)
        print(f"{pdf.name}: {n} images")
        total += n
    print(f"done: {total} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
