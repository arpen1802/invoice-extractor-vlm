"""FastAPI service for invoice -> structured JSON extraction.

Endpoints:
  GET  /health           -> {"status": "ok", "backend": ...}
  POST /extract  (image) -> schema-valid Invoice JSON (see serving/schema.py)

The heavy model backend is loaded once at startup (see serving/backends.py) and
selected via env vars, so this file stays about HTTP concerns only. Output is
always validated against the Pydantic schema before it leaves the process.

Run:
    INVOICE_BACKEND=vllm uvicorn serving.app:app --host 0.0.0.0 --port 8000
    INVOICE_BACKEND=mock uvicorn serving.app:app          # no GPU, for testing
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from serving.schema import Invoice

app = FastAPI(title="Invoice Extractor", version="1.0.0")
_backend = None


def _get_backend():
    """Lazy singleton so `import serving.app` never loads a model (imports stay
    cheap for tests); the model is created on first use / startup event."""
    global _backend
    if _backend is None:
        from serving.backends import get_backend
        _backend = get_backend()
    return _backend


@app.on_event("startup")
def _warm():
    # In production (vllm/hf) load the model at startup so the first request
    # isn't slow. Skip for mock to keep tests instant.
    if os.environ.get("INVOICE_BACKEND", "vllm").lower() != "mock":
        _get_backend()


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.environ.get("INVOICE_BACKEND", "vllm")}


@app.post("/extract", response_model=Invoice)
async def extract(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, f"expected an image, got {file.content_type}")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "empty file")
    try:
        result = _get_backend().extract(image_bytes)
    except ValueError as e:
        # model produced unparseable / schema-invalid output
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"inference error: {type(e).__name__}: {e}")
    # response_model=Invoice re-validates on the way out.
    return result
