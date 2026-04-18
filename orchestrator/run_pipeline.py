"""End-to-end pipeline client: image -> OCR -> compress -> decompress -> assert.

Owner: Sampreeth

Usage:
    python -m orchestrator.run_pipeline \\
        --image path/to/FontLre_Noisec_TE.png \\
        --ocr-url http://localhost:8001 \\
        --huffman-url http://localhost:8002

Steps:
    1. POST image to OCR service  -> extracted_text
    2. POST text  to /compress    -> payload_b64 + metrics
    3. POST b64   to /decompress  -> recovered_text
    4. Assert recovered_text == extracted_text  (lossless round-trip)
    5. Print summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run:  pip install requests")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def ocr_image(image_path: Path, ocr_url: str, noise_type: str | None) -> dict:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {}
        if noise_type:
            data["noise_type"] = noise_type
        resp = requests.post(f"{ocr_url}/ocr", files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def compress_text(text: str, huffman_url: str, algo: str) -> dict:
    resp = requests.post(
        f"{huffman_url}/compress",
        json={"text": text, "algo": algo},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def decompress_payload(payload_b64: str, huffman_url: str, algo: str) -> dict:
    resp = requests.post(
        f"{huffman_url}/decompress",
        json={"payload_b64": payload_b64, "algo": algo},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    image_path: Path,
    ocr_url: str,
    huffman_url: str,
    algo: str,
    noise_type: str | None,
    verbose: bool,
) -> dict:
    t_start = time.perf_counter()

    # Step 1 — OCR
    print(f"[1/4] OCR  {image_path.name} ...")
    ocr_result = ocr_image(image_path, ocr_url, noise_type)
    extracted_text: str = ocr_result["text"]
    ocr_latency: float = ocr_result.get("latency_ms", 0.0)
    print(f"      {len(extracted_text)} chars  ({ocr_latency:.1f} ms)")
    if verbose:
        print(f"      preview: {extracted_text[:80]!r}")

    # Step 2 — Compress
    print(f"[2/4] Compress (algo={algo}) ...")
    comp_result = compress_text(extracted_text, huffman_url, algo)
    payload_b64: str = comp_result["payload_b64"]
    comp_latency: float = comp_result.get("latency_ms", 0.0)
    print(
        f"      bits={comp_result['bits']}  ratio={comp_result['ratio']:.3f}"
        f"  entropy={comp_result['entropy']:.3f}  efficiency={comp_result['efficiency']:.3f}"
        f"  ({comp_latency:.1f} ms)"
    )

    # Step 3 — Decompress
    print("[3/4] Decompress ...")
    decomp_result = decompress_payload(payload_b64, huffman_url, algo)
    recovered_text: str = decomp_result["text"]
    decomp_latency: float = decomp_result.get("latency_ms", 0.0)
    print(f"      {len(recovered_text)} chars  ({decomp_latency:.1f} ms)")

    # Step 4 — Assert lossless
    print("[4/4] Lossless check ...")
    if recovered_text != extracted_text:
        print("FAIL  Round-trip mismatch!")
        if verbose:
            for i, (a, b) in enumerate(zip(extracted_text, recovered_text)):
                if a != b:
                    print(f"      first diff at char {i}: {a!r} vs {b!r}")
                    break
        sys.exit(1)
    print("      OK — round-trip lossless")

    total_ms = (time.perf_counter() - t_start) * 1000
    summary = {
        "image": image_path.name,
        "noise_type": noise_type,
        "algo": algo,
        "chars": len(extracted_text),
        "ocr_latency_ms": ocr_latency,
        "compress_latency_ms": comp_latency,
        "decompress_latency_ms": decomp_latency,
        "total_latency_ms": round(total_ms, 2),
        "compression_ratio": comp_result["ratio"],
        "entropy": comp_result["entropy"],
        "encoding_efficiency": comp_result["efficiency"],
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: OCR + Huffman compress")
    parser.add_argument("--image", required=True, help="Path to noisy PNG image")
    parser.add_argument("--ocr-url", default="http://localhost:8001", help="OCR service base URL")
    parser.add_argument("--huffman-url", default="http://localhost:8002", help="Huffman service base URL")
    parser.add_argument("--algo", default="fgk", choices=["fgk", "vitter"], help="Compression algorithm")
    parser.add_argument("--noise-type", default=None, choices=["f", "w", "c", "p"],
                        help="Noise type hint sent to OCR service")
    parser.add_argument("--verbose", action="store_true", help="Print text previews and diff on failure")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    run(
        image_path=image_path,
        ocr_url=args.ocr_url.rstrip("/"),
        huffman_url=args.huffman_url.rstrip("/"),
        algo=args.algo,
        noise_type=args.noise_type,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
