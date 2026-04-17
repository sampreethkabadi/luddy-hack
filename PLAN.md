# 2-Stage Neural Compression Pipeline — Team Plan

**Team:** Sampreeth · Anuj · Poornam
**Timeline:** 24 hours
**Track:** Graduate (all "optional" items are required)
**Stack:** PyTorch + FastAPI, Colab GPU for training
**Dataset:** `SimulatedNoisyOffice/` — 216 noisy images (540×420 grayscale) paired with 54 clean ground-truth images. 4 built-in noise types: folded sheets (f), wrinkled sheets (w), coffee stains (c), footprints (p). Splits encoded in filename: TR / VA / TE.
**Ground rule:** No AI-generated code in the final repo. Use Claude/ChatGPT only for planning, debugging, and concept clarification — write every line yourselves.

> **Important reframe:** This is *line-level OCR on printed English text*, not digit classification. Noise profiles are already provided (exceeds the "2+" grad requirement). The dataset has clean/noisy *image pairs* — no text labels. We generate text labels ourselves (see §2 Block A).

---

## 1. Role Assignments

| Member | Role | Primary ownership | Secondary support |
|---|---|---|---|
| **Anuj** | Software Dev + DLS expert | Stage 1 CNN (architecture, training, checkpoints), both FastAPI microservice skeletons | Review Huffman bit-layout for correctness |
| **Sampreeth** | Generalist DS | Stage 2 adaptive Huffman (FGK + bit I/O + round-trip tests), orchestrator client, dataset/noise pipeline | CNN data loaders, glue code |
| **Poornam** | Analyst expert | All metrics (per-profile CER, compression ratio, entropy, efficiency), latency benchmarks, CNN architecture diagram, README + 3-min pitch deck | Run ablation experiments, record demo |

**Rule of thumb:** If two people could own a task, the one with spare capacity picks it up. Ping in the team chat every ~3 hours with a status line.

---

## 2. Hour-by-Hour Timeline

### Block A — Kickoff, labels, scaffolding (Hours 0–3)
- **All:** 30-min design meeting. Confirm contracts below. Create the GitHub repo and branch model (`main`, `feature/*`). Decide CNN architecture route (see §3).
- **Anuj:** Repo scaffold (folders, `pyproject.toml`/`requirements.txt`, lint config, `.gitignore`). Stub FastAPI apps that return 200 on `/health`.
- **Sampreeth:** Solve the **label problem** — we have clean images but no text transcriptions. Three options, pick one:
  - **(A) Tesseract on clean images:** run `pytesseract` once over `clean_images_grayscale/`, save `labels.json` keyed by font/size/emphasis. Fast, pragmatic.
  - **(B) Synthetic pair generator:** render text strings with the same 9 font variants using PIL/Pillow, apply synthetic versions of the 4 noise types. Unlimited training data.
  - **(C) Hybrid:** Tesseract labels for the 54 real pairs + synthetic augmentation for volume. Recommended.
  Also write the `DatasetSplitter` that parses filenames to get (font_variant, noise_type, split) tuples.
- **Poornam:** Set up Colab notebooks (one per experiment), shared Google Sheet for experiment tracking. Create a `REPORT.md` skeleton with placeholders for every metric (per-noise-type CER table has 4 rows now: f, w, c, p).

**Checkpoint (h3):** `/health` endpoints respond, `labels.json` exists for all clean images, dataset iterator yields `(noisy_image_tensor, text_label, noise_type)` tuples.

### Block B — Parallel build (Hours 3–11)
Three tracks running simultaneously.

#### Track 1 — CNN OCR (Anuj)
The task is line-level recognition of English text. Before training, **split each 540×420 image into text lines** using horizontal projection profile (sum of dark pixels per row → trough detection). Target line crop: ~540×40 grayscale.

Run **3 architectures in parallel Colab notebooks**:
- **A. Pure CNN + CTC:** 4–5 conv blocks, reshape feature map to (sequence_len × features), linear projection to alphabet + blank, CTC loss. Smallest, fastest — strong baseline.
- **B. CRNN + CTC:** CNN feature extractor + 1–2 BiLSTM layers + CTC. Industry-standard line OCR. Best accuracy likely.
- **C. Denoiser (U-Net) → OCR (A or B):** two-headed model. Loss = α·reconstruction MSE (vs clean image) + β·CTC. Leverages the clean/noisy pairs the dataset is actually designed for. Strongest innovation lever.

For each: train on augmented data, log per-noise-type validation CER (4 rows: f, w, c, p). Gate: ≥95% char-level accuracy on clean val set.

**Alphabet:** lowercase + uppercase + digits + punctuation found in clean images + space + CTC blank (~95 chars). Emit the actual set from Tesseract output.

#### Track 2 — Huffman (Sampreeth)
Implement adaptive Huffman by hand:
1. **`BitWriter` / `BitReader`** — byte-buffered, handles partial bytes with a final-bits header so the decoder knows where to stop.
2. **FGK algorithm** (Faller–Gallager–Knuth) — pick FGK first because it's the cleanest to implement. Maintain the sibling-property tree, NYT node, symbol→node map.
3. Round-trip test: randomised strings + ASCII corpora + sample OCR output → assert `decompress(compress(x)) == x` byte-for-byte.
4. Stretch after MVP works: also implement **Vitter's algorithm** as a pluggable `--algo vitter` switch — great comparison slide for the pitch.

#### Track 3 — Metrics & pipeline eval (Poornam)
- `metrics.py` with: CER (Levenshtein-based), compression ratio (`len(input_bytes) / len(output_bytes)`), Shannon entropy of source text, encoding efficiency (`avg_bits_per_symbol / entropy`).
- A `benchmark.py` harness that runs N samples end-to-end and logs latency per stage and total.
- **Per-noise-type breakdown:** because the dataset already splits 4 noise types by filename, metrics must report CER separately for f / w / c / p (+ the overall aggregate).
- Draft architecture diagram in Excalidraw / draw.io (placeholder until CNN is chosen).

**Checkpoint (h11):** CNN arch chosen and hitting ≥95% on clean val. Huffman round-trip passes on a 10k-string stress test. Metrics module works standalone with the 4-row CER table.

### Block C — Integration (Hours 11–15)
- **Anuj:** Wire chosen CNN into `POST /ocr` (multipart image → `{text, per_char_confidence, latency_ms}`). Include the line-splitting preprocessing step server-side. Wire Huffman into `POST /compress` and `POST /decompress` (JSON in, base64 bytes out + metrics).
- **Sampreeth:** Write the orchestrator/client script: image file → OCR call → Compress call → Decompress call → assert lossless. Add CLI flags for noise type (f/w/c/p) and split.
- **Poornam:** Run the full pipeline on held-out TE set for each of the 4 noise types. Fill in the metrics tables in `REPORT.md`.

**Checkpoint (h15):** One command on a fresh machine runs the full demo on any TE image.

### Block D — Grad extras & polish (Hours 15–19)
- **Anuj:** Export CNN arch diagram (torchviz or draw by hand). Write the "design justification" section — kernel sizes, depth, activations, why CTC, why BiLSTM (if CRNN), parameter count trade-off, how the denoiser head helps noise robustness.
- **Sampreeth:** If the `RealNoisyOffice` folder is available, evaluate on it for qualitative real-world results (mentioned in dataset README). Package model weights and a download script.
- **Poornam:** Finalize README — setup, reproduce-training instructions, metrics tables (4 noise types × CER), latency report, architecture diagram embed. Start the 3-minute deck.

**Checkpoint (h19):** Repo passes a dry run on a teammate's machine from a clean `git clone`.

### Block E — Demo recording + pitch (Hours 19–22)
- **Poornam + Anuj:** Record screen demo — noisy input image → extracted text → compressed bytes → decompressed output that matches. Keep it tight, ~90 seconds of footage.
- **Poornam:** 3-minute deck targeting the rubric:
  - Slide 1: Problem + pipeline diagram
  - Slide 2: CNN architecture & design choices (10% innovation)
  - Slide 3: Huffman implementation highlights (FGK vs Vitter comparison)
  - Slide 4: Results table — per-profile CER, compression ratio, entropy, efficiency, latency
  - Slide 5: Real-world impact + what we'd extend
- **Sampreeth:** Rehearse the live demo fallback (in case video fails).

### Block F — Buffer + submission (Hours 22–24)
- Fix last-minute bugs, tighten README, tag release, submit repo link. **Do not start new features in this block.**

---

## 3. Experiment Matrix (for the "experiment as much as possible" goal)

| Axis | Variants to try | Owner | Decision gate |
|---|---|---|---|
| CNN architecture | Pure CNN+CTC · CRNN+CTC · U-Net denoiser → CRNN+CTC | Anuj | h9 pick winner by val CER aggregated across 4 noise types |
| Label source | Tesseract on clean · Synthetic render · Hybrid | Sampreeth | h3 pick; hybrid is the default recommendation |
| Line segmentation | Horizontal projection profile · Fixed equal slices · Learned (connected components) | Anuj | Ablation table — projection profile is safe default |
| Adaptive Huffman algo | FGK (required) · Vitter (stretch) | Sampreeth | FGK by h9; Vitter only if ahead of schedule |
| Augmentation | Real 4 noise types only · + synthetic Gaussian/S&P/blur · + random scale/rotation | Anuj | Whichever clears 95% first wins |
| Denoiser loss weight α | {0, 0.1, 0.5, 1.0} (if arch C) | Anuj + Poornam | Ablation table in report — great for justification section |

Log every run into the shared sheet with: commit hash, seed, config, metric values. This directly feeds Poornam's analysis slide.

---

## 4. Service Contracts

Lock these in hour 0 so Anuj and Sampreeth don't block each other.

### Stage 1 — OCR
- `POST /ocr`
  - Request: multipart `image` (PNG) + optional `noise_type` string (`f` | `w` | `c` | `p` | unknown)
  - Response JSON: `{ "text": "There exist several methods to design\nto be filled in...", "lines": ["There exist...", "to be filled..."], "latency_ms": 124.0 }`
- `GET /health` → `{"status": "ok"}`

### Stage 2 — Compression
- `POST /compress`
  - Request JSON: `{ "text": "7413", "algo": "fgk" }`
  - Response JSON: `{ "payload_b64": "...", "bits": 27, "ratio": 1.18, "entropy": 2.1, "efficiency": 0.97, "latency_ms": 0.3 }`
- `POST /decompress`
  - Request JSON: `{ "payload_b64": "...", "algo": "fgk" }`
  - Response JSON: `{ "text": "7413", "latency_ms": 0.2 }`

Keep the contract in `docs/api.md` and treat it as the source of truth.

---

## 5. Repository Layout (target)

```
luddy_hack/
├── README.md                  # Poornam
├── PLAN.md                    # this file
├── REPORT.md                  # metrics + diagrams (Poornam)
├── docs/
│   ├── api.md                 # service contracts
│   └── architecture.png       # CNN diagram
├── data/
│   ├── dataset.py             # NoisyOffice loader + filename parser (Sampreeth)
│   ├── labels.py              # Tesseract/synthetic label generator (Sampreeth)
│   ├── lines.py               # line segmentation (projection profile) (Anuj)
│   └── labels.json            # generated
├── stage1_ocr/
│   ├── model.py               # CRNN / U-Net+CRNN (Anuj)
│   ├── ctc.py                 # CTC decode (greedy + beam) (Anuj)
│   ├── train.py
│   ├── service.py             # FastAPI app
│   └── weights/
├── stage2_huffman/
│   ├── bitio.py               # BitWriter / BitReader (Sampreeth)
│   ├── fgk.py                 # adaptive Huffman
│   ├── vitter.py              # stretch
│   └── service.py             # FastAPI app
├── orchestrator/
│   └── run_pipeline.py        # end-to-end client
├── benchmarks/
│   ├── latency.py
│   └── results/
└── notebooks/
    ├── 01_lenet.ipynb
    ├── 02_resnet.ipynb
    └── 03_denoiser.ipynb
```

---

## 6. Deliverables Checklist (from the rubric)

- [ ] Stage 1 source + trained weights + reproduce-training instructions
- [ ] Stage 2 source (FGK from scratch, no zlib/gzip)
- [ ] Decompressor that round-trips losslessly (verified by test)
- [ ] README with setup + usage
- [ ] 2+ noise profiles, per-profile accuracy reported
- [ ] Compression ratio + entropy + efficiency reported
- [ ] End-to-end latency benchmark in README
- [ ] CNN architecture diagram + design justification
- [ ] Recorded demo video
- [ ] 3-minute pitch deck
- [ ] Submit GitHub repo link via portal

---

## 7. Risk Register

| Risk | Owner | Mitigation |
|---|---|---|
| CNN stuck below 95% on noisy val | Anuj | Fall back to Pure CNN+CTC with heavy synthetic-data pretraining; don't chase exotic archs |
| FGK bit-layout bug causes decompress mismatch | Sampreeth | Round-trip test first, implementation second; test on adversarial inputs (empty, single char, huge repeats, unicode) |
| Tesseract labels are dirty/wrong | Sampreeth | Manually spot-check 10 of 54; if quality is bad, switch to synthetic-render path or hand-correct |
| Dataset too small (only 216 noisy images) | Anuj + Sampreeth | Synthetic augmentation is mandatory; generate 10k+ synthetic line images with the 4 noise signatures |
| Colab GPU disconnects mid-training | Anuj | Checkpoint every epoch to Drive; train in ≤30-min chunks |
| Line segmentation misses lines / merges them | Anuj | Tune projection-profile threshold on clean images first; fall back to fixed 11-line equal slicing |
| Scope creep on Vitter | Sampreeth | Hard gate: only start after FGK round-trip passes AND h15 checkpoint clears |
| Video recording fails close to deadline | Poornam | Record a backup at h19, polish at h21 |

---

## 8. Communication Cadence

- Async standup every 3 hours: one line — "done / doing / blocked."
- Any contract change needs a ping in chat before merging.
- Merge only via PR with at least a skim from another teammate. No direct pushes to `main` after hour 10.
