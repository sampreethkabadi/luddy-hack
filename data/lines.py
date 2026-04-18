"""Line segmentation for NoisyOffice page images.

Owner: Anuj (preprocessing for the CNN)

Approach: horizontal projection profile.
  1. Binarize the grayscale image (dark pixels = ink = 1).
  2. Sum dark pixels per row → 1-D profile.
  3. Smooth with a moving-average kernel to suppress noise.
  4. Find contiguous rows where ink density exceeds a threshold → text bands.
  5. Crop each band and resize to a fixed line height for the CNN.

Fallback: equal-height slicing when the projection profile yields an
implausible number of lines (< MIN_LINES or > MAX_LINES).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Expected number of text lines per page (sanity gate for fallback).
_MIN_LINES = 5
_MAX_LINES = 20
_DEFAULT_TARGET_H = 40   # pixels — matches the CNN input height
_DEFAULT_SMOOTH_K = 7    # moving-average kernel width (rows)
_DEFAULT_INK_FRAC = 0.04 # row must have ≥ 4 % of max-profile value to be "text"
_DEFAULT_MIN_GAP = 4     # min consecutive ink rows to count as a line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_lines(
    image: Image.Image | np.ndarray,
    target_height: int = _DEFAULT_TARGET_H,
    smooth_k: int = _DEFAULT_SMOOTH_K,
    ink_frac: float = _DEFAULT_INK_FRAC,
    min_gap: int = _DEFAULT_MIN_GAP,
) -> list[Image.Image]:
    """Split a page image into individual text-line crops.

    Args:
        image: PIL Image (any mode) or numpy array (H, W) uint8.
        target_height: Output line height in pixels (CNN input height).
        smooth_k: Moving-average kernel width for the projection profile.
        ink_frac: Fraction of the peak profile value used as the ink threshold.
        min_gap: Minimum consecutive ink-rows needed to start a new line band.

    Returns:
        List of PIL Images, each with height == target_height, in top-to-bottom
        order.  Never empty: falls back to equal-height slicing if the profile
        method is unstable.
    """
    pil = _to_pil(image)
    arr = np.array(pil)           # (H, W) uint8, 0=black 255=white

    crops = _projection_crops(arr, smooth_k, ink_frac, min_gap)

    # Sanity check — fall back to equal slicing if result is implausible.
    if not (_MIN_LINES <= len(crops) <= _MAX_LINES):
        crops = _equal_slice_crops(arr, n_slices=11)  # 11 ~ avg lines per page

    return [_resize_line(c, target_height) for c in crops]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("L")
    return image.convert("L")


def _projection_crops(
    arr: np.ndarray,
    smooth_k: int,
    ink_frac: float,
    min_gap: int,
) -> list[np.ndarray]:
    """Return list of row-band arrays using the projection-profile method."""
    # Dark pixels = ink (value < 128).
    binary = (arr < 128).astype(np.float32)          # (H, W)
    profile = binary.sum(axis=1)                      # (H,)

    # Smooth.
    kernel = np.ones(smooth_k) / smooth_k
    smooth = np.convolve(profile, kernel, mode="same")

    threshold = smooth.max() * ink_frac
    is_ink = smooth > threshold

    # Collect contiguous ink bands.
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for r, ink in enumerate(is_ink):
        if ink and not in_band:
            start = r
            in_band = True
        elif not ink and in_band:
            in_band = False
            if r - start >= min_gap:
                bands.append((start, r))
    if in_band and len(arr) - start >= min_gap:
        bands.append((start, len(arr)))

    return [arr[r0:r1, :] for r0, r1 in bands]


def _equal_slice_crops(arr: np.ndarray, n_slices: int) -> list[np.ndarray]:
    """Divide the image into n_slices equal horizontal bands."""
    H = arr.shape[0]
    step = max(H // n_slices, 1)
    slices = [arr[i : i + step, :] for i in range(0, H, step)]
    return [s for s in slices if s.shape[0] > 0]


def _resize_line(band: np.ndarray, target_height: int) -> Image.Image:
    """Resize a line crop to target_height while preserving aspect ratio."""
    pil = Image.fromarray(band)
    w, h = pil.size
    if h == 0:
        h = 1
    new_w = max(int(w * target_height / h), 1)
    return pil.resize((new_w, target_height), Image.LANCZOS)
