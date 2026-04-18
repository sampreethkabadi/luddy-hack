"""FastAPI OCR microservice — OCRNet (character-level) edition.

Owner: Anuj

Pipeline:
    uploaded image
        → line segmentation  (data/lines.py)
        → per-line character segmentation  (data/chars.py)
        → OCRNet inference on each 28×28 character crop
        → join characters → join lines → return text

Endpoints:
    GET  /health
    POST /ocr    multipart: image (PNG) + optional noise_type
                 → { text, lines, latency_ms, backend }

Weights: stage1_ocr/weights/best.pt  (or $OCR_WEIGHTS env var)
Fallback: Tesseract if weights file is absent.

Run with:
    uvicorn stage1_ocr.service:app --port 8001 --reload
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from data.lines import segment_lines
from data.chars import segment_chars
from stage1_ocr.model import OCRNet, EMNIST_LABELS

app = FastAPI(title="OCR Service", version="2.0.0")

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

_WEIGHTS_PATH  = Path(os.getenv("OCR_WEIGHTS", "stage1_ocr/weights/best.pt"))
_model: OCRNet | None = None
_device        = torch.device("cpu")
_use_tesseract = False
_to_tensor     = transforms.ToTensor()


@app.on_event("startup")
def _load_model() -> None:
    global _model, _device, _use_tesseract

    if _WEIGHTS_PATH.exists():
        ckpt    = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        n_cls   = ckpt.get("num_classes", 62)
        _model  = OCRNet(num_classes=n_cls)
        _model.load_state_dict(ckpt["state"])
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device).eval()
        acc = ckpt.get("accuracy", "?")
        print(f"[OCR] OCRNet loaded from {_WEIGHTS_PATH}  "
              f"(acc={acc}, device={_device})")
    else:
        print(f"[OCR] No weights at {_WEIGHTS_PATH} — falling back to Tesseract.")
        _use_tesseract = True


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _classify_char(crop: Image.Image) -> str:
    """Run one 28×28 crop through OCRNet → character string."""
    tensor = _to_tensor(crop).unsqueeze(0).to(_device)   # (1, 1, 28, 28)
    logits = _model(tensor)                               # (1, 62)
    idx    = logits.argmax(dim=1).item()
    return EMNIST_LABELS[idx]


def _infer_cnn(pil_image: Image.Image) -> tuple[str, list[str]]:
    """Full pipeline: page → lines → chars → OCRNet → text."""
    line_crops = segment_lines(pil_image, target_height=28)
    decoded_lines: list[str] = []

    for line_crop in line_crops:
        char_crops = segment_chars(line_crop, char_size=28)
        line_text  = "".join(_classify_char(c) for c in char_crops)
        if line_text:
            decoded_lines.append(line_text)

    return "\n".join(decoded_lines), decoded_lines


def _infer_tesseract(pil_image: Image.Image) -> tuple[str, list[str]]:
    """Tesseract fallback used before best.pt is available."""
    try:
        import pytesseract
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="No weights found and pytesseract is not installed.",
        )
    raw   = pytesseract.image_to_string(pil_image, lang="eng",
                                        config="--psm 6 --oem 3")
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    return "\n".join(lines), lines


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "backend": "tesseract" if _use_tesseract else "ocr_net",
    }


@app.post("/ocr")
async def ocr(
    image:      UploadFile       = File(...),
    noise_type: str | None       = Form(default=None),
):
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode uploaded image.")

    t0 = time.perf_counter()
    if _use_tesseract:
        text, lines = _infer_tesseract(pil_image)
    else:
        text, lines = _infer_cnn(pil_image)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "text":       text,
        "lines":      lines,
        "latency_ms": latency_ms,
        "noise_type": noise_type,
        "backend":    "tesseract" if _use_tesseract else "ocr_net",
    }
