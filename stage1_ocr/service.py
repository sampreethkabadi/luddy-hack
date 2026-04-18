"""FastAPI OCR microservice.

Owner: Anuj

Endpoints:
    GET  /health
    POST /ocr    multipart: image (PNG) + optional noise_type
                 -> { text, lines, latency_ms }

Startup: loads weights from stage1_ocr/weights/best.pt (or $OCR_WEIGHTS).
Fallback: if no weights file is found, falls back to Tesseract so the
          orchestrator can run end-to-end before training is complete.

Run with:
    uvicorn stage1_ocr.service:app --port 8001 --reload
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from data.lines import segment_lines
from stage1_ocr.ctc import Alphabet, build_alphabet, greedy_decode
from stage1_ocr.model import build_model

app = FastAPI(title="OCR Service", version="1.0.0")

# ---------------------------------------------------------------------------
# Startup — model loading
# ---------------------------------------------------------------------------

_WEIGHTS_PATH = Path(os.getenv("OCR_WEIGHTS", "stage1_ocr/weights/best.pt"))
_LABELS_PATH  = Path("data/labels.json")

# Populated on startup.
_model:    torch.nn.Module | None = None
_alphabet: Alphabet | None        = None
_device:   torch.device           = torch.device("cpu")
_use_tesseract: bool              = False


@app.on_event("startup")
def _load_model() -> None:
    global _model, _alphabet, _device, _use_tesseract

    if _WEIGHTS_PATH.exists():
        ckpt = torch.load(_WEIGHTS_PATH, map_location="cpu")
        _alphabet = Alphabet(chars=ckpt["alphabet"])
        _model = build_model(ckpt.get("arch", "crnn_ctc"), num_classes=_alphabet.size)
        _model.load_state_dict(ckpt["state"])
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device).eval()
        print(f"[OCR] Model loaded from {_WEIGHTS_PATH}  (arch={ckpt.get('arch','crnn_ctc')}, device={_device})")
    else:
        print(
            f"[OCR] WARNING: no weights at {_WEIGHTS_PATH} — falling back to Tesseract.\n"
            "       Train the model and place best.pt there to enable CNN inference."
        )
        _use_tesseract = True
        if _LABELS_PATH.exists():
            _alphabet = build_alphabet(_LABELS_PATH)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_cnn(pil_image: Image.Image) -> tuple[str, list[str]]:
    """Run CNN inference: segment lines → model → CTC decode → join."""
    from torchvision import transforms

    assert _model is not None and _alphabet is not None

    to_tensor = transforms.ToTensor()
    line_crops = segment_lines(pil_image)

    decoded_lines: list[str] = []
    for crop in line_crops:
        tensor = to_tensor(crop).unsqueeze(0).to(_device)   # (1, 1, 40, W)

        with torch.no_grad():
            if hasattr(_model, "forward_ocr"):
                # UNetDenoiserCRNN: run denoiser on full image first, but
                # for per-line inference we skip the denoiser here.
                log_probs = _model.forward_ocr(tensor)
            else:
                log_probs = _model(tensor)                   # (T, 1, C)

        text, _ = greedy_decode(log_probs[:, 0, :], _alphabet)
        decoded_lines.append(text)

    full_text = "\n".join(decoded_lines)
    return full_text, decoded_lines


def _infer_tesseract(pil_image: Image.Image) -> tuple[str, list[str]]:
    """Tesseract fallback (used before model weights are available)."""
    try:
        import pytesseract
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="No model weights and pytesseract is not installed. Cannot perform OCR.",
        )
    raw = pytesseract.image_to_string(pil_image, lang="eng", config="--psm 6 --oem 3")
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    return "\n".join(lines), lines


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "tesseract" if _use_tesseract else "cnn",
    }


@app.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    noise_type: str | None = Form(default=None),
):
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    t0 = time.perf_counter()

    if _use_tesseract:
        text, lines = _infer_tesseract(pil_image)
    else:
        text, lines = _infer_cnn(pil_image)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "text": text,
        "lines": lines,
        "latency_ms": latency_ms,
        "noise_type": noise_type,
        "backend": "tesseract" if _use_tesseract else "cnn",
    }
