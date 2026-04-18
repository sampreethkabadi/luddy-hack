"""Character segmentation for text-line crops.

Owner: Anuj

Approach: vertical projection profile.
  1. Binarize the line crop (dark pixels = ink).
  2. Sum dark pixels per column → 1-D profile.
  3. Find contiguous ink columns → character bounding boxes.
  4. Pad each crop to a square, resize to 28×28 (EMNIST input size).

Fallback: fixed equal-width slicing when too few or too many characters
are detected relative to the expected character count.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_CHAR_SIZE   = 28       # EMNIST input resolution
_MIN_WIDTH   = 3        # columns — narrower blobs are noise
_SMOOTH_K    = 3        # moving-average kernel for the column profile
_INK_FRAC    = 0.04     # fraction of peak profile used as ink threshold
_MERGE_GAP   = 2        # merge adjacent ink regions separated by ≤ this many columns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_chars(
    line_image: Image.Image | np.ndarray,
    char_size: int = _CHAR_SIZE,
    min_width: int = _MIN_WIDTH,
    smooth_k: int = _SMOOTH_K,
    ink_frac: float = _INK_FRAC,
    merge_gap: int = _MERGE_GAP,
) -> list[Image.Image]:
    """Split a text-line crop into individual character images.

    Args:
        line_image: PIL Image or numpy array (H, W) uint8.
        char_size:  Output square size in pixels (default 28 for EMNIST).
        min_width:  Minimum column width to count as a character.
        smooth_k:   Moving-average kernel width for the column profile.
        ink_frac:   Fraction of peak profile value used as ink threshold.
        merge_gap:  Adjacent ink regions separated by ≤ this are merged.

    Returns:
        List of char_size × char_size PIL Images (grayscale), one per
        detected character.  Never empty: falls back to equal-width
        slicing if the projection method yields no characters.
    """
    pil = _to_pil(line_image)
    arr = np.array(pil)

    boxes = _projection_boxes(arr, smooth_k, ink_frac, min_width, merge_gap)

    if not boxes:
        boxes = _equal_slice_boxes(arr)

    return [_crop_and_resize(pil, x0, x1, char_size) for x0, x1 in boxes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("L")
    return image.convert("L")


def _projection_boxes(
    arr: np.ndarray,
    smooth_k: int,
    ink_frac: float,
    min_width: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    """Return (x0, x1) column-span pairs for each character."""
    binary  = (arr < 128).astype(np.float32)       # ink = 1
    profile = binary.sum(axis=0)                   # (W,)

    kernel = np.ones(smooth_k) / smooth_k
    smooth = np.convolve(profile, kernel, mode="same")

    threshold = max(smooth.max() * ink_frac, 1e-6)
    is_ink    = smooth > threshold

    # Collect raw ink spans.
    spans: list[tuple[int, int]] = []
    in_span = False
    start   = 0
    for c, ink in enumerate(is_ink):
        if ink and not in_span:
            start   = c
            in_span = True
        elif not ink and in_span:
            in_span = False
            if c - start >= min_width:
                spans.append((start, c))
    if in_span and arr.shape[1] - start >= min_width:
        spans.append((start, arr.shape[1]))

    # Merge spans that are very close together (connected components).
    if not spans:
        return []
    merged: list[tuple[int, int]] = [spans[0]]
    for x0, x1 in spans[1:]:
        px0, px1 = merged[-1]
        if x0 - px1 <= merge_gap:
            merged[-1] = (px0, x1)
        else:
            merged.append((x0, x1))

    return merged


def _equal_slice_boxes(arr: np.ndarray, n_chars: int = 10) -> list[tuple[int, int]]:
    """Divide the line into n_chars equal-width columns as fallback."""
    W    = arr.shape[1]
    step = max(W // n_chars, 1)
    return [(i, min(i + step, W)) for i in range(0, W, step) if i + step <= W + 1]


def _crop_and_resize(pil: Image.Image, x0: int, x1: int, size: int) -> Image.Image:
    """Crop a column span, pad to square, resize to size×size."""
    crop = pil.crop((x0, 0, x1, pil.height))
    w, h = crop.size
    sq   = max(w, h, 1)
    padded = Image.new("L", (sq, sq), 255)          # white background
    padded.paste(crop, ((sq - w) // 2, (sq - h) // 2))
    return padded.resize((size, size), Image.LANCZOS)
