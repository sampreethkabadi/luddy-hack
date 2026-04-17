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
