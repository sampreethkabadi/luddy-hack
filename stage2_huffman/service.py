"""FastAPI compression microservice.

Owner: Sampreeth

Endpoints (see docs/api.md):
- GET  /health
- POST /compress   -- text -> payload_b64 + metrics (ratio, entropy, efficiency)
- POST /decompress -- payload_b64 -> text

Dispatches to fgk.py (default) or vitter.py (if algo="vitter") based on the
request body. Computes compression metrics alongside the payload.

Not yet implemented.
"""
