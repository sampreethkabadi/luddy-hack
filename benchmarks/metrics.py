
"""
metrics.py
Poornam — Luddy Hack, Stage 1+2 evaluation

Four metrics:
  1. CER          — character error rate (Levenshtein / len(reference))
  2. compression_ratio — len(original_bytes) / len(compressed_bytes)
  3. shannon_entropy   — H(X) = -sum(p * log2(p)) in bits/symbol
  4. encoding_efficiency — entropy / avg_bits_per_symbol (1.0 = perfect)

All functions are pure (no side-effects, no file I/O) so they are
easy to unit-test and call from benchmark.py or the FastAPI services.
"""

import math
from collections import Counter
from typing import Union


# ─────────────────────────────────────────────
#  1. Character Error Rate (CER)
# ─────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """
    Standard dynamic-programming Levenshtein distance.
    Counts the minimum edits (insert, delete, substitute) to turn a → b.
    Space: O(min(len(a), len(b)))  — single-row optimisation.
    """
    # Always iterate over the longer string in the outer loop
    # so the 'prev' row is the shorter one → less memory.
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))          # base case: delete all of b

    for i, ch_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)           # base case: delete i chars of a
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                curr[j] = prev[j - 1]       # characters match → no edit
            else:
                curr[j] = 1 + min(
                    prev[j],                # deletion
                    curr[j - 1],            # insertion
                    prev[j - 1],            # substitution
                )
        prev = curr

    return prev[len(b)]


def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = Levenshtein(ref, hyp) / len(ref).

    Returns a float in [0, ∞).
      0.0  → perfect match
      1.0  → as many errors as characters in the reference
      >1.0 → more errors than characters (hypothesis is very wrong)

    Args:
        reference:  ground-truth text string
        hypothesis: OCR-predicted text string

    Raises:
        ValueError: if reference is empty (CER is undefined)
    """
    if len(reference) == 0:
        raise ValueError("Reference string is empty — CER is undefined.")
    distance = _levenshtein(reference, hypothesis)
    return distance / len(reference)


def cer_accuracy(reference: str, hypothesis: str) -> float:
    """Convenience wrapper: returns 1 - CER, clamped to [0, 1]."""
    return max(0.0, 1.0 - cer(reference, hypothesis))


# ─────────────────────────────────────────────
#  2. Compression Ratio
# ─────────────────────────────────────────────

def compression_ratio(original: Union[str, bytes],
                      compressed: bytes) -> float:
    """
    Compression ratio = original_size / compressed_size.

    > 1.0 → compressed is smaller  (good)
    = 1.0 → no change
    < 1.0 → compressed is larger   (happens on short / random input)

    Args:
        original:   original text (str) or raw bytes
        compressed: bytes produced by the Huffman compressor

    Raises:
        ValueError: if compressed payload is empty
    """
    if isinstance(original, str):
        original_bytes = original.encode("utf-8")
    else:
        original_bytes = original

    if len(compressed) == 0:
        raise ValueError("Compressed payload is empty.")

    return len(original_bytes) / len(compressed)


# ─────────────────────────────────────────────
#  3. Shannon Entropy
# ─────────────────────────────────────────────

def shannon_entropy(text: str) -> float:
    """
    Shannon entropy H(X) of a text string, measured in bits/symbol.

    H(X) = -Σ  p(x) · log₂(p(x))    for all symbols x in the alphabet

    This is the theoretical lower bound on average bits-per-symbol
    achievable by any lossless code over this source.

    Args:
        text: input string (treat each character as a symbol)

    Returns:
        entropy in bits/symbol.  0.0 if text has only one unique symbol.

    Raises:
        ValueError: if text is empty
    """
    if len(text) == 0:
        raise ValueError("Input text is empty — entropy is undefined.")

    counts = Counter(text)
    n = len(text)

    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)   # -p·log2(p) is always >= 0

    return entropy


# ─────────────────────────────────────────────
#  4. Encoding Efficiency
# ─────────────────────────────────────────────

def encoding_efficiency(text: str, compressed_bits: int) -> float:
    """
    Encoding efficiency = H(X) / avg_bits_per_symbol.

    Where avg_bits_per_symbol = compressed_bits / len(text).

    Interpretation:
      1.0  → the code achieves the theoretical Shannon limit (perfect)
      0.9  → 10% overhead above the entropy lower bound
      <0.5 → the code is quite inefficient

    Args:
        text:            original text that was compressed
        compressed_bits: total number of bits in the compressed output
                         (get this from the Huffman service response field "bits")

    Raises:
        ValueError: if text is empty or compressed_bits is 0
    """
    if len(text) == 0:
        raise ValueError("Input text is empty.")
    if compressed_bits <= 0:
        raise ValueError("compressed_bits must be a positive integer.")

    h = shannon_entropy(text)
    avg_bits = compressed_bits / len(text)

    if avg_bits == 0:
        raise ValueError("avg_bits_per_symbol is zero — cannot compute efficiency.")

    return h / avg_bits


# ─────────────────────────────────────────────
#  5. Per-noise-type aggregation helper
# ─────────────────────────────────────────────

NOISE_TYPES = ("f", "w", "c", "p")   # folded, wrinkled, coffee, footprints

def aggregate_cer_by_noise(
    results: list[dict]
) -> dict:
    """
    Given a list of per-sample result dicts, compute mean CER per noise type
    and an overall aggregate.

    Each dict must have keys:
        "noise_type": one of "f", "w", "c", "p"
        "reference":  ground-truth string
        "hypothesis": OCR output string

    Returns a dict like:
        {
            "f": {"mean_cer": 0.04, "mean_accuracy": 0.96, "n": 12},
            "w": {...},
            "c": {...},
            "p": {...},
            "overall": {"mean_cer": 0.05, "mean_accuracy": 0.95, "n": 48},
        }
    """
    buckets: dict[str, list[float]] = {t: [] for t in NOISE_TYPES}

    for r in results:
        noise = r["noise_type"]
        c = cer(r["reference"], r["hypothesis"])
        if noise in buckets:
            buckets[noise].append(c)

    output = {}
    all_cers = []

    for noise_type, cers_list in buckets.items():
        if cers_list:
            mean = sum(cers_list) / len(cers_list)
            output[noise_type] = {
                "mean_cer": round(mean, 4),
                "mean_accuracy": round(1 - mean, 4),
                "n": len(cers_list),
            }
            all_cers.extend(cers_list)
        else:
            output[noise_type] = {"mean_cer": None, "mean_accuracy": None, "n": 0}

    if all_cers:
        overall_mean = sum(all_cers) / len(all_cers)
        output["overall"] = {
            "mean_cer": round(overall_mean, 4),
            "mean_accuracy": round(1 - overall_mean, 4),
            "n": len(all_cers),
        }

    return output


# ─────────────────────────────────────────────
#  6. Quick self-test (run this cell in Colab)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- CER ---
    ref  = "the cat sat on the mat"
    hyp1 = "the cat sat on the mat"   # perfect
    hyp2 = "the cat sat on the map"   # 1 substitution
    hyp3 = "cat sat on mat"           # 3 deletions
    print("=== CER ===")
    print(f"  perfect match:      {cer(ref, hyp1):.4f}  (expect 0.0000)")
    print(f"  1 substitution:     {cer(ref, hyp2):.4f}  (expect ~0.0455)")
    print(f"  3 deletions:        {cer(ref, hyp3):.4f}  (expect ~0.1818)")

    # --- Compression ratio ---
    original_text = "AABBBCCCC"
    fake_compressed = b"\xA3\x1F"    # pretend 2-byte output
    ratio = compression_ratio(original_text, fake_compressed)
    print(f"\n=== Compression Ratio ===")
    print(f"  9 bytes → 2 bytes:  {ratio:.2f}x  (expect 4.50x)")

    # --- Shannon entropy ---
    print(f"\n=== Shannon Entropy ===")
    print(f"  'AABBBCCCC':  {shannon_entropy('AABBBCCCC'):.4f} bits/sym")
    print(f"  'AAAA':       {shannon_entropy('AAAA'):.4f} bits/sym  (expect 0.0)")
    print(f"  'ABCD':       {shannon_entropy('ABCD'):.4f} bits/sym  (expect 2.0)")

    # --- Encoding efficiency ---
    text = "AABBBCCCC"
    # Optimal code for this dist would be ~1.53 bits/sym
    # If our Huffman used 15 bits total for 9 symbols → 1.67 bits/sym
    eff = encoding_efficiency(text, compressed_bits=15)
    print(f"\n=== Encoding Efficiency ===")
    print(f"  15 bits for 9 symbols: {eff:.4f}  (expect ~0.917)")

    # --- Aggregate by noise type ---
    print(f"\n=== Aggregate CER by noise type ===")
    sample_results = [
        {"noise_type": "f", "reference": "hello world", "hypothesis": "hello world"},
        {"noise_type": "f", "reference": "hello world", "hypothesis": "helo world"},
        {"noise_type": "w", "reference": "test line",   "hypothesis": "test line"},
        {"noise_type": "c", "reference": "coffee stain","hypothesis": "coffe stain"},
        {"noise_type": "p", "reference": "footprint",   "hypothesis": "footprint"},
    ]
    table = aggregate_cer_by_noise(sample_results)
    for noise, stats in table.items():
        print(f"  {noise}: {stats}")
