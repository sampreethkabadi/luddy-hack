from __future__ import annotations

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from data.lines import segment_lines
from data.chars import segment_chars
from stage1_ocr.model import OCRNet, EMNIST_LABELS

app = FastAPI(title="OCR Service", version="3.0.0")

_WEIGHTS_PATH  = Path(os.getenv("OCR_WEIGHTS", "stage1_ocr/weights/best.pt"))
_model: OCRNet | None = None
_device        = torch.device("cpu")
_has_model     = False
_has_tesseract = False
_to_tensor     = transforms.ToTensor()


@app.on_event("startup")
def _load() -> None:
    global _model, _device, _has_model, _has_tesseract

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _has_tesseract = True
    except Exception:
        pass

    if _WEIGHTS_PATH.exists():
        ckpt   = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        _model = OCRNet(num_classes=ckpt.get("num_classes", 62))
        _model.load_state_dict(ckpt["state"])
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device).eval()
        _has_model = True
        print(f"[OCR] loaded  acc={ckpt.get('accuracy','?')}  device={_device}")

    print(f"[OCR] backend: {_active_backend()}")


def _active_backend() -> str:
    if _has_model and _has_tesseract:
        return "hybrid"
    if _has_model:
        return "cnn_only"
    if _has_tesseract:
        return "tesseract_only"
    return "unavailable"


@torch.no_grad()
def _classify_char(crop: Image.Image) -> str:
    tensor = _to_tensor(crop).unsqueeze(0).to(_device)
    idx    = _model(tensor).argmax(dim=1).item()
    return EMNIST_LABELS[idx]


def _classify_word_crop(word_crop: Image.Image) -> str:
    char_crops = segment_chars(word_crop, char_size=28)
    return "".join(_classify_char(c) for c in char_crops)


def _infer_hybrid(pil: Image.Image) -> tuple[str, list[str]]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(
        pil, lang="eng",
        config="--psm 6 --oem 3",
        output_type=Output.DICT,
    )

    line_map: dict[tuple, list[dict]] = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = str(data["conf"][i])
        if not word or conf == "-1":
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        line_map.setdefault(key, []).append({
            "word_num": data["word_num"][i],
            "left":     data["left"][i],
            "top":      data["top"][i],
            "width":    data["width"][i],
            "height":   data["height"][i],
        })

    decoded_lines: list[str] = []
    for key in sorted(line_map.keys()):
        words = sorted(line_map[key], key=lambda w: w["word_num"])
        parts: list[str] = []
        for w in words:
            l, t, wd, ht = w["left"], w["top"], w["width"], w["height"]
            if wd < 2 or ht < 2:
                continue
            parts.append(_classify_word_crop(pil.crop((l, t, l + wd, t + ht))))
        if parts:
            decoded_lines.append(" ".join(parts))

    if not decoded_lines:
        return _infer_cnn_only(pil)

    return "\n".join(decoded_lines), decoded_lines


def _infer_cnn_only(pil: Image.Image) -> tuple[str, list[str]]:
    line_crops = segment_lines(pil, target_height=28)
    decoded_lines: list[str] = []
    for line_crop in line_crops:
        text = _classify_word_crop(line_crop)
        if text:
            decoded_lines.append(text)
    return "\n".join(decoded_lines), decoded_lines


def _infer_tesseract(pil: Image.Image) -> tuple[str, list[str]]:
    import pytesseract
    raw   = pytesseract.image_to_string(pil, lang="eng", config="--psm 6 --oem 3")
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    return "\n".join(lines), lines


@app.get("/health")
def health():
    return {"status": "ok", "backend": _active_backend()}


@app.post("/ocr")
async def ocr(
    image:      UploadFile  = File(...),
    noise_type: str | None  = Form(default=None),
):
    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image.")

    t0 = time.perf_counter()
    backend = _active_backend()

    if backend == "unavailable":
        raise HTTPException(status_code=503, detail="No model weights and no Tesseract installed.")
    elif backend == "hybrid":
        text, lines = _infer_hybrid(pil)
    elif backend == "cnn_only":
        text, lines = _infer_cnn_only(pil)
    else:
        text, lines = _infer_tesseract(pil)

    return {
        "text":       text,
        "lines":      lines,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "noise_type": noise_type,
        "backend":    backend,
    }
