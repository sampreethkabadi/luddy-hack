"""
benchmark.py
Poornam — Luddy Hack, end-to-end latency + metrics harness

Runs N samples through the full pipeline:
    noisy image → POST /ocr → POST /compress → POST /decompress → assert lossless

Outputs:
  - Per-sample rows (noise_type, cer, ratio, entropy, efficiency, latency_ocr_ms,
                     latency_compress_ms, latency_decompress_ms, latency_e2e_ms)
  - Summary table aggregated by noise type
  - JSON results file saved to benchmarks/results/

Usage (Colab or terminal):
    python benchmark.py --image_dir data/SimulatedNoisyOffice/test \
                        --labels    data/labels.json \
                        --ocr_url   http://localhost:8001 \
                        --comp_url  http://localhost:8002 \
                        --n         50 \
                        --algo      fgk \
                        --out       benchmarks/results/run_01.json

All HTTP calls use requests. Install with: pip install requests
"""

import argparse
import base64
import json
import math
import os
import statistics
import time
from pathlib import Path

import requests

from metrics import (
    cer,
    aggregate_cer_by_noise,
    compression_ratio,
    shannon_entropy,
    encoding_efficiency,
)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

NOISE_TYPES = ("f", "w", "c", "p")
NOISE_LABELS = {"f": "Folded", "w": "Wrinkled", "c": "Coffee", "p": "Footprints"}

# ─────────────────────────────────────────────
#  1. Filename parser  (mirrors dataset.py logic)
# ─────────────────────────────────────────────

def parse_filename(filename: str) -> dict:
    """
    NoisyOffice filenames encode split and noise type, e.g.:
        SimulatedNoisyOffice_TE_f_001.png  → split=TE, noise=f, idx=001
        SimulatedNoisyOffice_VA_w_012.png  → split=VA, noise=w, idx=012

    Returns dict with keys: split, noise_type, idx
    Returns noise_type="unknown" if pattern doesn't match.
    """
    stem = Path(filename).stem          # drop extension
    parts = stem.split("_")
    try:
        # expect [..., SPLIT, NOISETYPE, IDX]
        idx        = parts[-1]
        noise_type = parts[-2].lower()
        split      = parts[-3].upper()
        if noise_type not in NOISE_TYPES:
            noise_type = "unknown"
        return {"split": split, "noise_type": noise_type, "idx": idx}
    except IndexError:
        return {"split": "unknown", "noise_type": "unknown", "idx": "000"}


# ─────────────────────────────────────────────
#  2. HTTP helpers
# ─────────────────────────────────────────────

def call_ocr(image_path: str, noise_type: str, ocr_url: str) -> dict:
    """
    POST /ocr   →  { text, lines, latency_ms }
    Returns the full response dict plus a wall-clock latency measured client-side.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    t0 = time.perf_counter()
    resp = requests.post(
        f"{ocr_url}/ocr",
        files={"image": ("image.png", image_bytes, "image/png")},
        data={"noise_type": noise_type},
        timeout=30,
    )
    client_latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    result = resp.json()
    # Prefer server-reported latency if available, fall back to client-side
    result.setdefault("latency_ms", client_latency_ms)
    result["_client_latency_ms"] = client_latency_ms
    return result


def call_compress(text: str, algo: str, comp_url: str) -> dict:
    """
    POST /compress  →  { payload_b64, bits, ratio, entropy, efficiency, latency_ms }
    """
    t0 = time.perf_counter()
    resp = requests.post(
        f"{comp_url}/compress",
        json={"text": text, "algo": algo},
        timeout=10,
    )
    client_latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    result = resp.json()
    result.setdefault("latency_ms", client_latency_ms)
    result["_client_latency_ms"] = client_latency_ms
    return result


def call_decompress(payload_b64: str, algo: str, comp_url: str) -> dict:
    """
    POST /decompress  →  { text, latency_ms }
    """
    t0 = time.perf_counter()
    resp = requests.post(
        f"{comp_url}/decompress",
        json={"payload_b64": payload_b64, "algo": algo},
        timeout=10,
    )
    client_latency_ms = (time.perf_counter() - t0) * 1000

    resp.raise_for_status()
    result = resp.json()
    result.setdefault("latency_ms", client_latency_ms)
    result["_client_latency_ms"] = client_latency_ms
    return result


# ─────────────────────────────────────────────
#  3. Single-sample runner
# ─────────────────────────────────────────────

def run_sample(
    image_path: str,
    reference: str,
    noise_type: str,
    ocr_url: str,
    comp_url: str,
    algo: str,
) -> dict:
    """
    Run one image through the full pipeline and return a result dict.
    Raises on HTTP error or lossless-roundtrip failure.
    """
    wall_start = time.perf_counter()

    # Stage 1 — OCR
    ocr_result = call_ocr(image_path, noise_type, ocr_url)
    hypothesis = ocr_result["text"]
    latency_ocr_ms = ocr_result["latency_ms"]

    # Stage 2a — Compress
    comp_result = call_compress(hypothesis, algo, comp_url)
    payload_b64      = comp_result["payload_b64"]
    compressed_bits  = comp_result["bits"]
    latency_comp_ms  = comp_result["latency_ms"]

    # Stage 2b — Decompress  +  lossless check
    decomp_result    = call_decompress(payload_b64, algo, comp_url)
    recovered_text   = decomp_result["text"]
    latency_decomp_ms = decomp_result["latency_ms"]

    if recovered_text != hypothesis:
        raise ValueError(
            f"Lossless roundtrip FAILED for {image_path}\n"
            f"  original : {repr(hypothesis[:80])}\n"
            f"  recovered: {repr(recovered_text[:80])}"
        )

    wall_e2e_ms = (time.perf_counter() - wall_start) * 1000

    # Metrics (computed locally — cross-check against service values)
    compressed_bytes = base64.b64decode(payload_b64)
    sample_cer        = cer(reference, hypothesis) if reference else None
    sample_ratio      = compression_ratio(hypothesis, compressed_bytes)
    sample_entropy    = shannon_entropy(hypothesis) if hypothesis else 0.0
    sample_efficiency = encoding_efficiency(hypothesis, compressed_bits) if hypothesis else 0.0

    return {
        "image":               os.path.basename(image_path),
        "noise_type":          noise_type,
        "reference":           reference,
        "hypothesis":          hypothesis,
        "cer":                 round(sample_cer, 4) if sample_cer is not None else None,
        "compression_ratio":   round(sample_ratio, 4),
        "entropy_bits":        round(sample_entropy, 4),
        "encoding_efficiency": round(sample_efficiency, 4),
        "compressed_bits":     compressed_bits,
        "latency_ocr_ms":      round(latency_ocr_ms, 2),
        "latency_compress_ms": round(latency_comp_ms, 2),
        "latency_decomp_ms":   round(latency_decomp_ms, 2),
        "latency_e2e_ms":      round(wall_e2e_ms, 2),
        "lossless":            True,
    }


# ─────────────────────────────────────────────
#  4. Summary table builder
# ─────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")

def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_summary(rows: list[dict]) -> dict:
    """
    Aggregate per-sample rows into a summary dict keyed by noise type + overall.

    Each bucket contains:
        n, mean_cer, mean_accuracy, mean_ratio, mean_entropy,
        mean_efficiency, mean_e2e_ms, stdev_e2e_ms, p95_e2e_ms
    """
    buckets: dict[str, dict[str, list]] = {}

    for row in rows:
        nt = row["noise_type"]
        if nt not in buckets:
            buckets[nt] = {k: [] for k in (
                "cer", "ratio", "entropy", "efficiency", "e2e_ms",
                "ocr_ms", "compress_ms", "decomp_ms"
            )}
        b = buckets[nt]
        if row["cer"] is not None:
            b["cer"].append(row["cer"])
        b["ratio"].append(row["compression_ratio"])
        b["entropy"].append(row["entropy_bits"])
        b["efficiency"].append(row["encoding_efficiency"])
        b["e2e_ms"].append(row["latency_e2e_ms"])
        b["ocr_ms"].append(row["latency_ocr_ms"])
        b["compress_ms"].append(row["latency_compress_ms"])
        b["decomp_ms"].append(row["latency_decomp_ms"])

    summary = {}
    all_rows_flat: dict[str, list] = {k: [] for k in buckets[next(iter(buckets))]} if buckets else {}

    for nt, b in buckets.items():
        mean_cer = _mean(b["cer"])
        e2e_sorted = sorted(b["e2e_ms"])
        p95_idx = max(0, math.ceil(0.95 * len(e2e_sorted)) - 1)

        summary[nt] = {
            "label":             NOISE_LABELS.get(nt, nt),
            "n":                 len(b["e2e_ms"]),
            "mean_cer":          round(mean_cer, 4),
            "mean_accuracy":     round(1 - mean_cer, 4),
            "mean_ratio":        round(_mean(b["ratio"]), 4),
            "mean_entropy":      round(_mean(b["entropy"]), 4),
            "mean_efficiency":   round(_mean(b["efficiency"]), 4),
            "mean_ocr_ms":       round(_mean(b["ocr_ms"]), 2),
            "mean_compress_ms":  round(_mean(b["compress_ms"]), 2),
            "mean_decomp_ms":    round(_mean(b["decomp_ms"]), 2),
            "mean_e2e_ms":       round(_mean(b["e2e_ms"]), 2),
            "stdev_e2e_ms":      round(_stdev(b["e2e_ms"]), 2),
            "p95_e2e_ms":        round(e2e_sorted[p95_idx], 2),
        }
        for k in all_rows_flat:
            all_rows_flat[k].extend(b[k])

    # Overall aggregate
    if all_rows_flat:
        all_cer  = all_rows_flat["cer"]
        all_e2e  = sorted(all_rows_flat["e2e_ms"])
        p95_idx  = max(0, math.ceil(0.95 * len(all_e2e)) - 1)
        mean_cer = _mean(all_cer)
        summary["overall"] = {
            "label":             "Overall",
            "n":                 len(all_e2e),
            "mean_cer":          round(mean_cer, 4),
            "mean_accuracy":     round(1 - mean_cer, 4),
            "mean_ratio":        round(_mean(all_rows_flat["ratio"]), 4),
            "mean_entropy":      round(_mean(all_rows_flat["entropy"]), 4),
            "mean_efficiency":   round(_mean(all_rows_flat["efficiency"]), 4),
            "mean_ocr_ms":       round(_mean(all_rows_flat["ocr_ms"]), 2),
            "mean_compress_ms":  round(_mean(all_rows_flat["compress_ms"]), 2),
            "mean_decomp_ms":    round(_mean(all_rows_flat["decomp_ms"]), 2),
            "mean_e2e_ms":       round(_mean(all_e2e), 2),
            "stdev_e2e_ms":      round(_stdev(all_e2e), 2),
            "p95_e2e_ms":        round(all_e2e[p95_idx], 2),
        }

    return summary


# ─────────────────────────────────────────────
#  5. Pretty-print to stdout (for README copy-paste)
# ─────────────────────────────────────────────

def print_summary_table(summary: dict) -> None:
    """Print a markdown-formatted results table to stdout."""
    order = [t for t in NOISE_TYPES if t in summary] + ["overall"]

    print("\n## Benchmark Results\n")

    # CER / accuracy table
    print("### OCR Accuracy by Noise Type\n")
    print(f"| Noise type   | N  | Mean CER | Mean Accuracy |")
    print(f"|---|---|---|---|")
    for nt in order:
        s = summary[nt]
        print(f"| {s['label']:<12} | {s['n']:<2} | {s['mean_cer']:.4f}   | {s['mean_accuracy']:.4f}        |")

    # Compression metrics table
    print("\n### Compression Metrics by Noise Type\n")
    print(f"| Noise type   | Ratio  | Entropy (bits/sym) | Efficiency |")
    print(f"|---|---|---|---|")
    for nt in order:
        s = summary[nt]
        print(f"| {s['label']:<12} | {s['mean_ratio']:.4f} | {s['mean_entropy']:.4f}             | {s['mean_efficiency']:.4f}     |")

    # Latency table
    print("\n### End-to-End Latency by Noise Type\n")
    print(f"| Noise type   | OCR (ms) | Compress (ms) | Decompress (ms) | E2E mean (ms) | E2E p95 (ms) |")
    print(f"|---|---|---|---|---|---|")
    for nt in order:
        s = summary[nt]
        print(
            f"| {s['label']:<12} | {s['mean_ocr_ms']:>8.2f} | {s['mean_compress_ms']:>13.2f} | "
            f"{s['mean_decomp_ms']:>15.2f} | {s['mean_e2e_ms']:>13.2f} | {s['p95_e2e_ms']:>12.2f} |"
        )


# ─────────────────────────────────────────────
#  6. Main entrypoint
# ─────────────────────────────────────────────

def run_benchmark(
    image_dir: str,
    labels_path: str,
    ocr_url: str,
    comp_url: str,
    n: int,
    algo: str,
    out_path: str,
    split_filter: str = "TE",
) -> dict:
    """
    Load images from image_dir, look up references from labels.json,
    run up to n samples, save results to out_path.
    """
    # Load labels
    with open(labels_path) as f:
        labels: dict = json.load(f)

    # Collect image paths matching split filter
    image_paths = sorted(Path(image_dir).glob("*.png"))
    filtered = []
    for p in image_paths:
        meta = parse_filename(p.name)
        if split_filter and meta["split"] != split_filter:
            continue
        filtered.append((str(p), meta))

    if not filtered:
        raise FileNotFoundError(
            f"No images found in {image_dir} with split={split_filter}"
        )

    # Cap at n
    filtered = filtered[:n]

    print(f"Running benchmark: {len(filtered)} samples | split={split_filter} | algo={algo}")
    print(f"  OCR service:         {ocr_url}")
    print(f"  Compression service: {comp_url}\n")

    rows = []
    failed = 0

    for i, (image_path, meta) in enumerate(filtered, 1):
        noise_type = meta["noise_type"]
        # Labels keyed by filename stem (set by Sampreeth's labels.py)
        stem = Path(image_path).stem
        reference = labels.get(stem, labels.get(Path(image_path).name, ""))

        try:
            row = run_sample(
                image_path=image_path,
                reference=reference,
                noise_type=noise_type,
                ocr_url=ocr_url,
                comp_url=comp_url,
                algo=algo,
            )
            rows.append(row)
            cer_str = f"{row['cer']:.4f}" if row["cer"] is not None else "n/a"
            print(
                f"  [{i:>3}/{len(filtered)}] {Path(image_path).name:<40} "
                f"noise={noise_type}  cer={cer_str}  "
                f"ratio={row['compression_ratio']:.2f}x  "
                f"e2e={row['latency_e2e_ms']:.1f}ms  ✓"
            )
        except Exception as e:
            failed += 1
            print(f"  [{i:>3}/{len(filtered)}] FAILED — {Path(image_path).name}: {e}")

    summary = build_summary(rows)

    output = {
        "config": {
            "image_dir":    image_dir,
            "labels_path":  labels_path,
            "ocr_url":      ocr_url,
            "comp_url":     comp_url,
            "algo":         algo,
            "split_filter": split_filter,
            "n_requested":  n,
            "n_completed":  len(rows),
            "n_failed":     failed,
        },
        "rows":    rows,
        "summary": summary,
    }

    # Save JSON
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    print_summary_table(summary)
    return output


# ─────────────────────────────────────────────
#  7. CLI
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="End-to-end pipeline benchmark")
    p.add_argument("--image_dir", default="data/SimulatedNoisyOffice/test",
                   help="Directory of noisy PNG images")
    p.add_argument("--labels",    default="data/labels.json",
                   help="Path to labels.json (stem → reference text)")
    p.add_argument("--ocr_url",   default="http://localhost:8001",
                   help="Base URL of the OCR microservice")
    p.add_argument("--comp_url",  default="http://localhost:8002",
                   help="Base URL of the compression microservice")
    p.add_argument("--n",         type=int, default=50,
                   help="Max number of samples to run")
    p.add_argument("--algo",      default="fgk", choices=["fgk", "vitter"],
                   help="Huffman algorithm variant")
    p.add_argument("--split",     default="TE",
                   help="Dataset split to use: TR / VA / TE")
    p.add_argument("--out",       default="benchmarks/results/run.json",
                   help="Output JSON path")
    return p.parse_args()


# ─────────────────────────────────────────────
#  8. Colab-friendly runner (call this directly in a notebook cell)
# ─────────────────────────────────────────────

def run_in_colab(
    image_dir:  str = "data/SimulatedNoisyOffice/test",
    labels_path:str = "data/labels.json",
    ocr_url:    str = "http://localhost:8001",
    comp_url:   str = "http://localhost:8002",
    n:          int = 50,
    algo:       str = "fgk",
    out_path:   str = "benchmarks/results/run.json",
    split:      str = "TE",
) -> dict:
    """
    Drop-in for Colab: no argparse, just call with keyword args.

    Example:
        from benchmark import run_in_colab
        results = run_in_colab(n=20, algo='fgk')
    """
    return run_benchmark(
        image_dir=image_dir,
        labels_path=labels_path,
        ocr_url=ocr_url,
        comp_url=comp_url,
        n=n,
        algo=algo,
        out_path=out_path,
        split_filter=split,
    )


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        image_dir=args.image_dir,
        labels_path=args.labels,
        ocr_url=args.ocr_url,
        comp_url=args.comp_url,
        n=args.n,
        algo=args.algo,
        out_path=args.out,
        split_filter=args.split,
    )
