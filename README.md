# Luddy Hack 2026 — 2-Stage Neural Compression Pipeline

> CNN-based OCR on noisy scanned documents → adaptive Huffman compression → lossless recovery.
> Two communicating microservices. Graduate track.

## Team
- Sampreeth — Huffman + orchestration + dataset pipeline
- Anuj — CNN OCR + microservice infra
- Poornam — metrics + benchmarks + presentation

See [`PLAN.md`](./PLAN.md) for the full plan.

## Dataset

This repo does **not** ship the dataset. Download the **SimulatedNoisyOffice** corpus and place it at `./SimulatedNoisyOffice/` (structure per `SimulatedNoisyOffice/README_NoisyOffice.txt`).

## Setup

_TBD — Poornam to fill._

```
pip install -r requirements.txt
```

## Running the pipeline

_TBD — one-command demo goes here._

## Results

See [`REPORT.md`](./REPORT.md) for CER per noise type, compression metrics, and latency.

## Repo layout

```
data/              — dataset loader, label generation, line segmentation
stage1_ocr/        — CNN OCR microservice (FastAPI)
stage2_huffman/    — Adaptive Huffman microservice (FastAPI)
orchestrator/      — End-to-end pipeline client
benchmarks/        — Latency + accuracy measurement harness
docs/              — API contracts, architecture diagram
```
