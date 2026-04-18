"""Compression quality metrics.

Owner: Poornam (formulas) + Sampreeth (integration)

All functions work on raw strings / byte sequences and are side-effect free.
"""

from __future__ import annotations

import math
from collections import Counter


def compression_ratio(source: str | bytes, compressed: bytes) -> float:
    """len(source_bytes) / len(compressed_bytes).  >1 means the output shrank."""
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    if len(compressed) == 0:
        return float("inf")
    return len(src_bytes) / len(compressed)


def shannon_entropy(source: str | bytes) -> float:
    """Shannon entropy in bits/symbol over the byte distribution of *source*."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    n = len(data)
    if n == 0:
        return 0.0
    counts = Counter(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def avg_bits_per_symbol(source: str | bytes, compressed: bytes) -> float:
    """(8 * len(compressed_bytes)) / len(source_bytes)."""
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    n = len(src_bytes)
    if n == 0:
        return 0.0
    return (8 * len(compressed)) / n


def encoding_efficiency(source: str | bytes, compressed: bytes) -> float:
    """shannon_entropy / avg_bits_per_symbol.  1.0 = optimal."""
    abps = avg_bits_per_symbol(source, compressed)
    if abps == 0.0:
        return 0.0
    return shannon_entropy(source) / abps


def compression_metrics(source: str, compressed: bytes) -> dict:
    """Return all four metrics in one dict (used by the FastAPI service)."""
    return {
        "ratio": round(compression_ratio(source, compressed), 4),
        "entropy": round(shannon_entropy(source), 4),
        "avg_bits_per_symbol": round(avg_bits_per_symbol(source, compressed), 4),
        "efficiency": round(encoding_efficiency(source, compressed), 4),
    }
