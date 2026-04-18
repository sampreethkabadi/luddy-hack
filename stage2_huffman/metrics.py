from __future__ import annotations

import math
from collections import Counter


def compression_ratio(source: str | bytes, compressed: bytes) -> float:
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    if len(compressed) == 0:
        return float("inf")
    return len(src_bytes) / len(compressed)


def shannon_entropy(source: str | bytes) -> float:
    data = source.encode("utf-8") if isinstance(source, str) else source
    n = len(data)
    if n == 0:
        return 0.0
    counts = Counter(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def avg_bits_per_symbol(source: str | bytes, compressed: bytes) -> float:
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    n = len(src_bytes)
    if n == 0:
        return 0.0
    total_bits = (len(compressed) - 1) * 8
    if len(compressed) > 1:
        meaningful_last = compressed[0] if compressed else 8
        total_bits = (len(compressed) - 2) * 8 + meaningful_last
    return total_bits / n


def encoding_efficiency(source: str | bytes, compressed: bytes) -> float:
    h = shannon_entropy(source)
    avg = avg_bits_per_symbol(source, compressed)
    if avg == 0:
        return 0.0
    return h / avg


def compression_metrics(source: str | bytes, compressed: bytes) -> dict:
    return {
        "ratio":      round(compression_ratio(source, compressed), 4),
        "entropy":    round(shannon_entropy(source), 4),
        "avg_bits":   round(avg_bits_per_symbol(source, compressed), 4),
        "efficiency": round(encoding_efficiency(source, compressed), 4),
    }
