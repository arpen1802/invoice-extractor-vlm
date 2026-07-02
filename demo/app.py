"""Gradio demo: upload an invoice image -> extracted JSON.

Two modes:
  - API mode (default): POSTs to a running serving/app.py at INVOICE_API_URL.
    Keeps the demo lightweight and mirrors the production path.
  - Local mode: set INVOICE_DEMO_LOCAL=1 to call the backend in-process (uses
    serving/backends.py; pick backend via INVOICE_BACKEND, e.g. hf for CPU).

Run:
    # against a running API
    INVOICE_API_URL=http://localhost:8000 python demo/app.py
    # fully local, CPU fallback model
    INVOICE_DEMO_LOCAL=1 INVOICE_BACKEND=hf python demo/app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

API_URL = os.environ.get("INVOICE_API_URL", "http://localhost:8000")
LOCAL = os.environ.get("INVOICE_DEMO_LOCAL") == "1"

_backend = None


def _extract_local(image_bytes: bytes) -> dict:
    global _backend
    if _backend is None:
        from serving.backends import get_backend
        _backend = get_backend()
    return _backend.extract(image_bytes)


def _extract_api(image_bytes: bytes) -> dict:
    import requests
    r = requests.post(f"{API_URL}/extract",
                      files={"file": ("invoice.png", image_bytes, "image/png")},
                      timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text}")
    return r.json()


def extract(image):
    if image is None:
        return {"error": "upload an invoice image"}
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    data = buf.getvalue()
    try:
        return _extract_local(data) if LOCAL else _extract_api(data)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


with gr.Blocks(title="Invoice Extractor") as demo:
    gr.Markdown("# Invoice Extractor\nUpload an invoice image; get schema-valid JSON.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Invoice")
        out = gr.JSON(label="Extracted JSON")
    gr.Button("Extract").click(extract, inputs=inp, outputs=out)
    mode = "local backend" if LOCAL else f"API @ {API_URL}"
    gr.Markdown(f"_Mode: {mode}_")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
