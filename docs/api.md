# Service Contracts

Source of truth for the two microservices. Any contract change requires a ping in the team chat before merging.

Run services locally:
```bash
uvicorn stage1_ocr.service:app   --port 8001 --reload   # OCR
uvicorn stage2_huffman.service:app --port 8002 --reload  # Compression
```

---

## Stage 1 — OCR Microservice

Base URL: `http://localhost:8001`

### `GET /health`

Response `200`:
```json
{ "status": "ok", "backend": "cnn" }
```
`backend` is `"cnn"` when `stage1_ocr/weights/best.pt` is loaded, `"tesseract"` otherwise.

---

### `POST /ocr`

Request: `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | PNG file | ✓ | Grayscale or RGB page image |
| `noise_type` | string | — | `f` \| `w` \| `c` \| `p` — hint only, does not change output |

Response `200`:
```json
{
  "text":       "There exist several methods to design\nto be filled in...",
  "lines":      ["There exist several methods to design", "to be filled in..."],
  "latency_ms": 124.0,
  "noise_type": "f",
  "backend":    "cnn"
}
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Full page transcription, lines joined by `\n` |
| `lines` | list[string] | Per-line transcriptions in top-to-bottom order |
| `latency_ms` | float | Wall-clock inference time (segmentation + model) |
| `noise_type` | string \| null | Echoed from request |
| `backend` | string | `"cnn"` or `"tesseract"` |

Error responses:

| Code | Meaning |
|---|---|
| `400` | Uploaded file could not be decoded as an image |
| `503` | No weights and pytesseract not installed |

---

## Stage 2 — Compression Microservice

Base URL: `http://localhost:8002`

### `GET /health`

Response `200`:
```json
{ "status": "ok" }
```

---

### `POST /compress`

Request JSON:

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✓ | UTF-8 text to compress |
| `algo` | string | — | `"fgk"` (default) or `"vitter"` |

Response `200`:
```json
{
  "payload_b64": "base64-encoded-bitstream",
  "bits":        1234,
  "ratio":       1.42,
  "entropy":     4.37,
  "efficiency":  0.96,
  "latency_ms":  3.2
}
```

| Field | Type | Description |
|---|---|---|
| `payload_b64` | string | Base64-encoded compressed bytes (pass verbatim to `/decompress`) |
| `bits` | int | Total bits in the compressed payload |
| `ratio` | float | `len(source_bytes) / len(compressed_bytes)` — >1 means size shrank |
| `entropy` | float | Shannon entropy of the source in bits/symbol |
| `efficiency` | float | `entropy / avg_bits_per_symbol` — 1.0 = theoretically optimal |
| `latency_ms` | float | Compression wall-clock time |

Error responses:

| Code | Meaning |
|---|---|
| `400` | Malformed request body |
| `501` | `algo=vitter` requested but not implemented |

---

### `POST /decompress`

Request JSON:

| Field | Type | Required | Description |
|---|---|---|---|
| `payload_b64` | string | ✓ | Base64 string from `/compress` response |
| `algo` | string | — | Must match the algo used at compress time |

Response `200`:
```json
{ "text": "...", "latency_ms": 2.1 }
```

Error responses:

| Code | Meaning |
|---|---|
| `400` | Invalid base64 |
| `422` | Decompression failed (wrong algo or corrupted payload) |

---

## End-to-end contract invariant

For any input text `T` and any `algo`:
```
decompress(compress(T, algo), algo) == T     # byte-for-byte lossless
```

Verified by `orchestrator/run_pipeline.py` on every image in the TE split.

---

## Full pipeline flow

```
image.png
    │
    ▼  POST /ocr
┌─────────────────┐
│  OCR Service    │  line segmentation → CNN → CTC decode
│  :8001          │
└────────┬────────┘
         │ text
         ▼  POST /compress
┌─────────────────┐
│  Compression    │  adaptive Huffman (FGK or Vitter)
│  Service :8002  │
└────────┬────────┘
         │ payload_b64
         ▼  POST /decompress
┌─────────────────┐
│  Compression    │  decode → assert recovered == text
│  Service :8002  │
└─────────────────┘
```
