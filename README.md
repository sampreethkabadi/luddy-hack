# Luddy Hack 2026 — 2-Stage Neural Compression Pipeline

> CNN-based OCR on noisy scanned documents → adaptive Huffman compression → lossless recovery.
> Two communicating microservices. Graduate track.

## Team
- Sampreeth — Adaptive Huffman (FGK + Vitter), orchestration, dataset pipeline
- Anuj — CNN OCR architecture, microservice infrastructure
- Poornam — metrics, benchmarks, evaluation, presentation

---

## Results Summary

| Metric | Value |
|---|---|
| OCRNet clean accuracy (EMNIST byclass, 62-class) | **87.28%** |
| OCRNet + Gaussian noise (σ=0.3) | 85.67% |
| OCRNet + Salt & Pepper (5%) | 87.06% |
| Compression ratio (FGK / Vitter) | **1.607x** |
| Encoding efficiency | 86.9% |
| Compress latency p50 / p95 | 4.3ms / 5.7ms |
| End-to-end p50 / p95 | ~130ms / ~210ms |
| Round-trip lossless | ✓ always |

Full results in [`REPORT.md`](./REPORT.md).

---

## Architecture

```
Noisy image (PNG)
    │
    ▼
┌─────────────────────────────┐
│  Stage 1 — OCR Microservice │  :8001
│                             │
│  Tesseract word bbox        │
│      → per-word char seg    │
│      → OCRNet (PyTorch CNN) │
│      → text output          │
└────────────┬────────────────┘
             │ extracted text
             ▼
┌─────────────────────────────┐
│  Stage 2 — Compression      │  :8002
│  Microservice               │
│                             │
│  Adaptive Huffman (FGK or   │
│  Vitter) → compressed bytes │
│  + decompress → lossless    │
└─────────────────────────────┘
```

---

## Setup

### Requirements

```bash
pip install torch torchvision fastapi uvicorn python-multipart pillow numpy requests
# Optional (improves OCR accuracy via hybrid mode):
brew install tesseract      # macOS
# apt install tesseract-ocr # Linux
pip install pytesseract
```

### Train the OCR model

```bash
python stage1_ocr/train.py
# Trains OCRNet on EMNIST byclass (62 classes: 0-9, A-Z, a-z)
# Downloads dataset automatically on first run (~500MB)
# Expected time: ~15 min GPU / ~60 min CPU
# Saves weights to: stage1_ocr/weights/best.pt
```

### Start the services

```bash
# Terminal 1 — OCR service
python -m uvicorn stage1_ocr.service:app --port 8001 --reload

# Terminal 2 — Compression service
python -m uvicorn stage2_huffman.service:app --port 8002 --reload
```

### Verify both are running

```bash
curl http://localhost:8001/health
# → {"status":"ok","backend":"hybrid"}   (or "cnn_only" without Tesseract)

curl http://localhost:8002/health
# → {"status":"ok"}
```

---

## Running the Pipeline

### Single image (end-to-end)

```bash
# OCR
curl -X POST http://localhost:8001/ocr \
  -F "image=@path/to/noisy_doc.png" \
  -F "noise_type=gaussian"

# Compress the returned text
curl -X POST http://localhost:8002/compress \
  -H "Content-Type: application/json" \
  -d '{"text": "your extracted text here", "algo": "fgk"}'

# Decompress (round-trip check)
curl -X POST http://localhost:8002/decompress \
  -H "Content-Type: application/json" \
  -d '{"payload_b64": "<base64 from compress>", "algo": "fgk"}'
```

### Full pipeline script

```bash
python orchestrator/run_pipeline.py --image path/to/doc.png
```

### Benchmark (latency + accuracy)

```bash
python benchmarks/latency.py \
  --image_dir data/SimulatedNoisyOffice/test \
  --labels data/labels.json \
  --n 50 --algo fgk \
  --out benchmarks/results/run.json
```

---

## Repo Layout

```
stage1_ocr/
  model.py        — OCRNet: 3-block CNN, 62-class EMNIST classifier
  train.py        — Training loop with Gaussian + S&P noise augmentation
  service.py      — FastAPI: GET /health, POST /ocr (hybrid/cnn/tesseract)
  weights/        — best.pt saved here (gitignored — run train.py to reproduce)

stage2_huffman/
  fgk.py          — FGK adaptive Huffman encode/decode
  vitter.py       — Vitter Algorithm V (leaf-preferring, provably optimal)
  bitio.py        — BitWriter / BitReader for packed bit streams
  metrics.py      — compression_ratio, shannon_entropy, encoding_efficiency
  service.py      — FastAPI: POST /compress, POST /decompress

data/
  lines.py        — Horizontal projection line segmentation
  chars.py        — Vertical projection character segmentation
  dataset.py      — NoisyOfficeDataset (PyTorch Dataset)
  labels.py       — Tesseract batch OCR → labels.json

orchestrator/
  run_pipeline.py — End-to-end CLI: image → OCR → compress → assert lossless

benchmarks/
  latency.py      — Full pipeline benchmark harness
  metrics.py      — CER, compression ratio, entropy, efficiency
```

---

## CNN Architecture

**OCRNet** — 3-block convolutional classifier

```
Input:   (1, 28, 28) grayscale character crop
Block 1: Conv2d(1→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → (32,14,14)
Block 2: Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → (64,7,7)
         → Dropout2d(0.25)
Block 3: Conv2d(64→128, 3×3) → BatchNorm → ReLU               → (128,7,7)
FC1:     Linear(6272→256) → BatchNorm → ReLU → Dropout(0.5)
FC2:     Linear(256→62)   → logits
```

Training: EMNIST byclass (697,932 samples) + 33% Gaussian / 33% S&P noise augmentation.

---

## Adaptive Huffman — Two Variants

| Property | FGK | Vitter |
|---|---|---|
| Invariant | Sibling property | Sibling + leaves rank above internals |
| Leader selection | Highest-order in block | Leaves only |
| Code optimality | Good | Provably optimal at every step |
| Tests passing | 502 | 300 |

Both share the same public API: `encode(text) -> bytes`, `decode(bytes) -> str`.
