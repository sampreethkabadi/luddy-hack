# Results Report

_Owner: Poornam. Replace all `—` placeholders with real numbers after the pipeline run._

---

## 1. OCR — Character Accuracy per Noise Type

**Model:** OCRNet (CNN, 62-class EMNIST character classification)
**Training data:** EMNIST `byclass` (697,932 samples) + noise augmentation (Gaussian + salt-and-pepper)
**Test data:** SimulatedNoisyOffice TE split, 4 noise profiles

| Noise Type     | Description           | Test Accuracy | Test CER | # Samples |
|---|---|---|---|---|
| Folded (f)     | Folded sheets         | —             | —        | —         |
| Wrinkled (w)   | Wrinkled sheets       | —             | —        | —         |
| Coffee (c)     | Coffee stains         | —             | —        | —         |
| Footprint (p)  | Footprint marks       | —             | —        | —         |
| **Aggregate**  | All noise types       | —             | —        | —         |

> Gate: aggregate CER ≤ 0.15 on NoisyOffice TE (EMNIST clean accuracy target ≥ 85 %).

**EMNIST clean/noise accuracy (from training run):**

| Condition          | Accuracy |
|---|---|
| Clean              | —        |
| Gaussian (σ=0.3)   | —        |
| Salt & Pepper (5%) | —        |

---

## 2. Compression Metrics (FGK vs Vitter)

Run on OCR output text from the full TE split.

| Metric                              | FGK   | Vitter |
|---|---|---|
| Avg compression ratio (src/compressed) | —  | —      |
| Avg source entropy (bits/symbol)    | —     | —      |
| Avg encoded bits per symbol         | —     | —      |
| Encoding efficiency (entropy/avg bits) | —  | —      |
| Round-trip lossless                 | ✓     | ✓      |

> `benchmarks/latency.py` produces these numbers — run it after the services are up.

---

## 3. End-to-End Latency

Hardware: — (CPU / GPU model)

| Stage                   | p50 (ms) | p95 (ms) |
|---|---|---|
| OCR (image → text)      | —        | —        |
| Compress (text → bytes) | —        | —        |
| Decompress (bytes → text) | —      | —        |
| **End-to-end total**    | —        | —        |

---

## 4. CNN Architecture

**Model:** OCRNet — 3-block CNN classifier

```
Input:   (1, 28, 28)  — grayscale character crop
Block 1: Conv2d(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → (32, 14, 14)
Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) → (64, 7, 7)
         → Dropout2d(0.25)
Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU               → (128, 7, 7)
Flatten: 128×7×7 = 6272
FC1:     Linear(6272→256) → BN → ReLU → Dropout(0.5)
FC2:     Linear(256→62)   → logits
```

| Property         | Value                          |
|---|---|
| Input shape      | (1, 28, 28)                    |
| Output classes   | 62 (EMNIST byclass)            |
| Kernel sizes     | 3×3 throughout                 |
| Activations      | ReLU                           |
| Regularisation   | BatchNorm + Dropout2d + Dropout |
| Parameters       | 1,715,454                      |
| Loss             | CrossEntropyLoss               |
| Optimizer        | Adam (lr=0.001)                |
| LR schedule      | ReduceLROnPlateau (×0.5, patience=2) |
| Epochs           | 10                             |
| Augmentation     | 33% clean / 33% Gaussian / 33% salt-and-pepper |

**Pipeline (inference):**
```
page image (540×420)
  ↓ data/lines.py   — horizontal projection profile → ~11 line crops (28px tall)
  ↓ data/chars.py   — vertical projection profile   → 28×28 character crops
  ↓ OCRNet.forward  — classify each crop → EMNIST index → character
  ↓ join chars → join lines → full page text
```

---

## 5. Adaptive Huffman — Algorithm Comparison

| Property              | FGK                          | Vitter                          |
|---|---|---|
| Invariant             | Sibling property             | Sibling + leaves rank above internals |
| Leader selection      | Highest-order in block       | Leaves only (for leaf updates)  |
| Code optimality       | Good                         | Provably optimal at every step  |
| Implementation        | `stage2_huffman/fgk.py`      | `stage2_huffman/vitter.py`      |
| Round-trip tests      | 502 pass                     | 300 pass                        |

---

## 6. Ablations

| Variant                | EMNIST Accuracy | Notes                            |
|---|---|---|
| OCRNet baseline (clean) | —              | No noise augmentation            |
| OCRNet + noise aug      | —              | 33% Gaussian + 33% S&P           |
| Pure CNN + CTC          | n/a            | Alternate arch (not trained here)|
| FGK compression         | —              | Compression ratio on TE text     |
| Vitter compression      | —              | Compression ratio on TE text     |
