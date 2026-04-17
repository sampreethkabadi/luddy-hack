"""Line segmentation for 540x420 NoisyOffice page images.

Owner: Anuj (preprocessing for the CNN)

Approach: horizontal projection profile.
- Binarize image (or threshold grayscale)
- Sum dark pixels per row -> 1D signal
- Smooth and detect troughs (low-ink rows) as line separators
- Crop each line region to a uniform height (e.g. ~40px) for the CNN input

Fallback: fixed equal-height slicing if projection profile is unstable on
heavily-degraded inputs (folded/coffee noise).

Not yet implemented.
"""
