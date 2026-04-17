# Service Contracts

Source of truth for the two microservices. Any contract change requires a ping in the team chat before merging.

---

## Stage 1 — OCR Microservice

Base URL: `http://localhost:8001`

### `GET /health`
Response: `{"status": "ok"}`

### `POST /ocr`
Request: `multipart/form-data`
- `image` (required): PNG file
- `noise_type` (optional): one of `f` | `w` | `c` | `p` | `unknown`

Response (`200 OK`):
```json
{
  "text": "There exist several methods to design\nto be filled in...",
  "lines": [
    "There exist several methods to design",
    "to be filled in..."
  ],
  "latency_ms": 124.0
}
```

---

## Stage 2 — Compression Microservice

Base URL: `http://localhost:8002`

### `GET /health`
Response: `{"status": "ok"}`

### `POST /compress`
Request JSON:
```json
{ "text": "...", "algo": "fgk" }
```
- `algo`: `"fgk"` (required) or `"vitter"` (stretch goal)

Response:
```json
{
  "payload_b64": "base64-encoded-bitstream",
  "bits": 1234,
  "ratio": 1.42,
  "entropy": 4.37,
  "efficiency": 0.96,
  "latency_ms": 3.2
}
```

### `POST /decompress`
Request JSON:
```json
{ "payload_b64": "...", "algo": "fgk" }
```

Response:
```json
{ "text": "...", "latency_ms": 2.1 }
```

---

## Contract invariant

For any input text `T`:
```
decompress(compress(T)) == T     (byte-for-byte)
```
This is asserted in the end-to-end test in `orchestrator/`.
