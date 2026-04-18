from __future__ import annotations

import numpy as np
from PIL import Image

_MIN_LINES        = 5
_MAX_LINES        = 20
_DEFAULT_TARGET_H = 40
_DEFAULT_SMOOTH_K = 7
_DEFAULT_INK_FRAC = 0.04
_DEFAULT_MIN_GAP  = 4


def segment_lines(
    image: Image.Image | np.ndarray,
    target_height: int = _DEFAULT_TARGET_H,
    smooth_k: int = _DEFAULT_SMOOTH_K,
    ink_frac: float = _DEFAULT_INK_FRAC,
    min_gap: int = _DEFAULT_MIN_GAP,
) -> list[Image.Image]:
    pil   = _to_pil(image)
    arr   = np.array(pil)
    crops = _projection_crops(arr, smooth_k, ink_frac, min_gap)
    if not (_MIN_LINES <= len(crops) <= _MAX_LINES):
        crops = _equal_slice_crops(arr, n_slices=11)
    return [_resize_line(c, target_height) for c in crops]


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("L")
    return image.convert("L")


def _projection_crops(arr, smooth_k, ink_frac, min_gap) -> list[np.ndarray]:
    binary    = (arr < 128).astype(np.float32)
    profile   = binary.sum(axis=1)
    kernel    = np.ones(smooth_k) / smooth_k
    smooth    = np.convolve(profile, kernel, mode="same")
    threshold = smooth.max() * ink_frac
    is_ink    = smooth > threshold

    bands: list[tuple[int, int]] = []
    in_band = False
    start   = 0
    for r, ink in enumerate(is_ink):
        if ink and not in_band:
            start   = r
            in_band = True
        elif not ink and in_band:
            in_band = False
            if r - start >= min_gap:
                bands.append((start, r))
    if in_band and len(arr) - start >= min_gap:
        bands.append((start, len(arr)))

    return [arr[r0:r1, :] for r0, r1 in bands]


def _equal_slice_crops(arr: np.ndarray, n_slices: int) -> list[np.ndarray]:
    H    = arr.shape[0]
    step = max(H // n_slices, 1)
    return [s for s in (arr[i:i + step, :] for i in range(0, H, step)) if s.shape[0] > 0]


def _resize_line(band: np.ndarray, target_height: int) -> Image.Image:
    pil  = Image.fromarray(band)
    w, h = pil.size
    if h == 0:
        h = 1
    new_w = max(int(w * target_height / h), 1)
    return pil.resize((new_w, target_height), Image.LANCZOS)
