"""Ground-truth text label generation for NoisyOffice.

Owner: Sampreeth

Strategy: hybrid.
- Run Tesseract once over SimulatedNoisyOffice/clean_images_grayscale/ to produce
  text labels keyed by font/size/emphasis/split. Persist to data/labels.json.
- Optionally render synthetic (clean image, text) pairs with matching fonts
  (PIL + 9 font variants) for training-data volume.
- Manually spot-check a subset of Tesseract outputs; correct any obvious errors
  in labels.json.

Outputs:
- data/labels.json: { "FontLre_TR": "There exist several methods...", ... }

Not yet implemented.
"""

"""
data/labels.py
Batch OCR over clean_images_grayscale/ → data/labels.json

Key schema:  {font_variant}_{split}  e.g. "Fontfre_TE"
  - "full"  : full transcription joined by \n
  - "lines" : list of non-empty lines (preserves internal blank lines via
              paragraph markers — see note below)

Usage:
    python data/labels.py                         # default paths
    python data/labels.py --img_dir path/to/imgs --out data/labels.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    sys.exit(
        "Missing dependencies. Run:\n"
        "  pip install pytesseract Pillow\n"
        "and make sure the Tesseract binary is installed."
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TESSERACT_CONFIG = r"--psm 6 --oem 3"   # block of text, LSTM engine
LANG = "eng"

# Filename pattern: FontXxx_Clean_TE.png  |  FontXxx_Noise3_VA.png
# We want: font_variant = "FontXxx", split = "TE"
FNAME_RE = re.compile(
    r"^(?P<font>[A-Za-z0-9]+)_(?:Clean|Noise[a-zA-Z]+)_(?P<split>TR|VA|TE)\.png$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image) -> Image.Image:
    """Light preprocessing that reliably helps Tesseract on clean scans."""
    image = ImageOps.grayscale(image)
    # Upscale only if the image is small (< 1000 px wide)
    w, h = image.size
    if w < 1000:
        image = image.resize((w * 2, h * 2), Image.LANCZOS)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def ocr(image: Image.Image) -> tuple[str, list[str]]:
    """
    Returns (full_text, lines).

    `lines` is a flat list of non-empty text lines. Empty lines (paragraph
    breaks, leading/trailing blanks) are dropped so that each entry in
    `lines` corresponds 1-to-1 with a line crop produced by the image-side
    line segmenter.
    """
    raw = pytesseract.image_to_string(image, lang=LANG, config=TESSERACT_CONFIG)

    # Keep only non-empty lines (drops leading, trailing, AND paragraph breaks).
    # Simpler for downstream: image line crops align 1-to-1 with label lines.
    all_lines = [l.rstrip() for l in raw.splitlines() if l.strip()]

    full = "\n".join(all_lines)
    return full, all_lines


def parse_filename(fname: str) -> tuple[str, str] | None:
    """
    Returns (font_variant, split) or None if the filename doesn't match.
    e.g. "Fontfre_Clean_TE.png" → ("Fontfre", "TE")
    """
    m = FNAME_RE.match(fname)
    if not m:
        return None
    return m.group("font"), m.group("split")


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def build_labels(img_dir: Path) -> dict:
    png_files = sorted(img_dir.glob("*.png"))
    if not png_files:
        sys.exit(f"No .png files found in {img_dir}")

    labels: dict = {}
    skipped: list[str] = []

    for png_path in png_files:
        parsed = parse_filename(png_path.name)
        if parsed is None:
            skipped.append(png_path.name)
            continue

        font_variant, split = parsed
        key = f"{font_variant}_{split}"

        print(f"  OCR → {png_path.name}  (key: {key})", end="", flush=True)

        image = Image.open(png_path)
        image = preprocess(image)
        full, lines = ocr(image)

        # If the same key appears more than once (shouldn't, but guard anyway)
        if key in labels:
            print(f"\n  ⚠️  Duplicate key {key!r} — skipping {png_path.name}")
            continue

        labels[key] = {"full": full, "lines": lines}
        print(f"  [{len(lines)} lines]")

    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} file(s) with unrecognised names:")
        for s in skipped:
            print(f"   {s}")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Batch OCR → labels.json")
    parser.add_argument(
        "--img_dir",
        default="SimulatedNoisyOffice/clean_images_grayscale",
        help="Directory containing the clean PNG images "
             "(default: SimulatedNoisyOffice/clean_images_grayscale/)",
    )
    parser.add_argument(
        "--out",
        default="data/labels.json",
        help="Output path for labels.json (default: data/labels.json)",
    )
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    out_path = Path(args.out)

    if not img_dir.exists():
        sys.exit(f"Image directory not found: {img_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {img_dir} …")
    labels = build_labels(img_dir)

    json.dump(labels, out_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n✅  {len(labels)} entries written to {out_path}")
    print("\nSpot-check targets (edit labels.json for any errors you find):")
    for key in list(labels)[:6]:
        first_line = labels[key]["lines"][0] if labels[key]["lines"] else "(empty)"
        print(f"  {key}: {first_line[:60]}")


if __name__ == "__main__":
    main()

