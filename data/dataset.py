"""NoisyOffice dataset loader.

Owner: Sampreeth

Responsibilities:
- Parse filenames of form Font{A}{B}{C}_Noise{D}_{EE}.png where
    A = font size (f | n | L)
    B = font type (t | s | r)
    C = emphasis  (e | m)
    D = noise type (f folded | w wrinkled | c coffee | p footprint) or "Clean"
    EE = split (TR | VA | TE)
- Pair each noisy image with its clean ground-truth counterpart (same A/B/C, same split).
- Yield (image_tensor, text_label, noise_type, split) tuples.
- Depend on labels.json (produced by data/labels.py) for text ground truth.

Not yet implemented.
"""
