from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Run: pip install requests")


def ocr_image(image_path: Path, ocr_url: str, noise_type: str | None) -> dict:
    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, "image/png")}
        data = {"noise_type": noise_type} if noise_type else {}
        resp = requests.post(f"{ocr_url}/ocr", files=files, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def compress_text(text: str, huffman_url: str, algo: str) -> dict:
    resp = requests.post(f"{huffman_url}/compress", json={"text": text, "algo": algo}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def decompress_payload(payload_b64: str, huffman_url: str, algo: str) -> dict:
    resp = requests.post(f"{huffman_url}/decompress", json={"payload_b64": payload_b64, "algo": algo}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run(image_path: Path, ocr_url: str, huffman_url: str, algo: str, noise_type: str | None, verbose: bool) -> dict:
    t_start = time.perf_counter()

    print(f"[1/4] OCR  {image_path.name} ...")
    ocr_result = ocr_image(image_path, ocr_url, noise_type)
    extracted_text: str = ocr_result["text"]
    print(f"      {len(extracted_text)} chars  ({ocr_result.get('latency_ms', 0):.1f} ms)")
    if verbose:
        print(f"      preview: {extracted_text[:80]!r}")

    print(f"[2/4] Compress (algo={algo}) ...")
    comp_result = compress_text(extracted_text, huffman_url, algo)
    payload_b64: str = comp_result["payload_b64"]
    print(f"      bits={comp_result['bits']}  ratio={comp_result['ratio']:.3f}  entropy={comp_result['entropy']:.3f}  efficiency={comp_result['efficiency']:.3f}  ({comp_result.get('latency_ms', 0):.1f} ms)")

    print("[3/4] Decompress ...")
    decomp_result = decompress_payload(payload_b64, huffman_url, algo)
    recovered_text: str = decomp_result["text"]
    print(f"      {len(recovered_text)} chars  ({decomp_result.get('latency_ms', 0):.1f} ms)")

    print("[4/4] Lossless check ...")
    if recovered_text != extracted_text:
        print("FAIL  Round-trip mismatch!")
        sys.exit(1)
    print("      OK")

    total_ms = (time.perf_counter() - t_start) * 1000
    summary = {
        "image":               image_path.name,
        "noise_type":          noise_type,
        "algo":                algo,
        "chars":               len(extracted_text),
        "ocr_latency_ms":      ocr_result.get("latency_ms", 0),
        "compress_latency_ms": comp_result.get("latency_ms", 0),
        "decompress_latency_ms": decomp_result.get("latency_ms", 0),
        "total_latency_ms":    round(total_ms, 2),
        "compression_ratio":   comp_result["ratio"],
        "entropy":             comp_result["entropy"],
        "encoding_efficiency": comp_result["efficiency"],
    }
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: OCR + Huffman compress")
    parser.add_argument("--image",       required=True)
    parser.add_argument("--ocr-url",     default="http://localhost:8001")
    parser.add_argument("--huffman-url", default="http://localhost:8002")
    parser.add_argument("--algo",        default="fgk", choices=["fgk", "vitter"])
    parser.add_argument("--noise-type",  default=None,  choices=["f", "w", "c", "p"])
    parser.add_argument("--verbose",     action="store_true")
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
