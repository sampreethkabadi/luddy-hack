"""OCR accuracy metrics.

Owner: Poornam

- character_error_rate(pred: str, truth: str) -> float
  Levenshtein-based: edit_distance(pred, truth) / max(len(truth), 1)
- per_noise_cer(predictions_by_noise: dict) -> dict
  Aggregates CER across noise types (f, w, c, p, aggregate)

Not yet implemented.
"""
