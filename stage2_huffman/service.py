"""FastAPI compression microservice.

Owner: Sampreeth

Endpoints:
    GET  /health
    POST /compress    { text, algo? }  ->  { payload_b64, bits, ratio, entropy, efficiency, latency_ms }
    POST /decompress  { payload_b64, algo? }  ->  { text, latency_ms }

Run with:
    uvicorn stage2_huffman.service:app --port 8002 --reload
"""

from __future__ import annotations

import base64
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .fgk import encode as fgk_encode, decode as fgk_decode
from .metrics import compression_metrics

app = FastAPI(title="Adaptive Huffman Compression Service", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CompressRequest(BaseModel):
    text: str
    algo: str = Field(default="fgk", pattern="^(fgk|vitter)$")


class CompressResponse(BaseModel):
    payload_b64: str
    bits: int
    ratio: float
    entropy: float
    efficiency: float
    latency_ms: float


class DecompressRequest(BaseModel):
    payload_b64: str
    algo: str = Field(default="fgk", pattern="^(fgk|vitter)$")


class DecompressResponse(BaseModel):
    text: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_codec(algo: str):
    if algo == "fgk":
        return fgk_encode, fgk_decode
    # Vitter is a stretch goal; import lazily so missing impl doesn't break FGK.
    try:
        from .vitter import encode as ve, decode as vd  # type: ignore[import]
        return ve, vd
    except (ImportError, AttributeError):
        raise HTTPException(
            status_code=501,
            detail="Vitter algorithm not yet implemented. Use algo='fgk'.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest):
    enc, _ = _get_codec(req.algo)

    t0 = time.perf_counter()
    compressed = enc(req.text)
    latency_ms = (time.perf_counter() - t0) * 1000

    metrics = compression_metrics(req.text, compressed)

    return CompressResponse(
        payload_b64=base64.b64encode(compressed).decode(),
        bits=len(compressed) * 8,
        ratio=metrics["ratio"],
        entropy=metrics["entropy"],
        efficiency=metrics["efficiency"],
        latency_ms=round(latency_ms, 3),
    )


@app.post("/decompress", response_model=DecompressResponse)
def decompress(req: DecompressRequest):
    _, dec = _get_codec(req.algo)

    try:
        compressed = base64.b64decode(req.payload_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload.")

    t0 = time.perf_counter()
    try:
        text = dec(compressed)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Decompression failed: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000

    return DecompressResponse(text=text, latency_ms=round(latency_ms, 3))
