"""FastAPI OCR microservice.

Owner: Anuj

Endpoints (see docs/api.md):
- GET  /health
- POST /ocr    -- accepts PNG upload, returns extracted text + per-line breakdown

On startup: load weights from stage1_ocr/weights/best.pt into the chosen model.

Not yet implemented.
"""
