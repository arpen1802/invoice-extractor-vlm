"""Inference backends for the invoice extractor service.

Three backends behind one interface so the FastAPI app doesn't care which runs:

- **vllm**  (production, GPU): vLLM engine with **guided JSON decoding** driven by
            the Invoice schema, so output is schema-shaped by construction.
- **hf**    (fallback): plain transformers generation; works on CPU (slow) or GPU.
            Lets the demo run cheaply without vLLM/GPU.
- **mock**  (tests/CI): returns a fixed schema-valid object; no model, no deps.

Select with env var INVOICE_BACKEND (default: vllm). Model dir via INVOICE_MODEL
(base id or a path to the merged fine-tuned model / adapter).
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Protocol

from serving.prompt import EXTRACT_PROMPT
from serving.schema import Invoice

DEFAULT_MODEL = os.environ.get("INVOICE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_TOKENS = int(os.environ.get("INVOICE_MAX_TOKENS", "1024"))


class Backend(Protocol):
    def extract(self, image_bytes: bytes) -> dict: ...


def _pil(image_bytes: bytes):
    from PIL import Image
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _validate(raw: str) -> dict:
    """Parse model text to schema-valid dict. Raises ValueError on failure so the
    API layer can return a clean 422."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return valid JSON: {e}")
    return json.loads(Invoice(**obj).model_dump_json())


class MockBackend:
    """Deterministic schema-valid output for tests — no model loaded."""

    def extract(self, image_bytes: bytes) -> dict:
        return json.loads(Invoice(
            invoice_number="MOCK-001", invoice_date="2024-01-01",
            vendor_name="Mock Vendor", currency="USD", total=0.0,
            line_items=[{"description": "mock", "quantity": 1,
                         "unit_price": 0, "amount": 0}],
        ).model_dump_json())


class VLLMBackend:
    """vLLM + guided JSON decoding. GPU. Loaded lazily so importing this module
    stays cheap on machines without vLLM."""

    def __init__(self, model: str = DEFAULT_MODEL, quantization: str | None = None):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.llm = LLM(model=model, trust_remote_code=True,
                       quantization=quantization,          # e.g. "awq" / None
                       limit_mm_per_prompt={"image": 1})
        guided = GuidedDecodingParams(json=Invoice.model_json_schema())
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS,
                                     guided_decoding=guided)

    def _prompt(self, image):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": EXTRACT_PROMPT},
        ]}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def extract(self, image_bytes: bytes) -> dict:
        img = _pil(image_bytes)
        out = self.llm.generate(
            {"prompt": self._prompt(img), "multi_modal_data": {"image": img}},
            self.params,
        )
        return _validate(out[0].outputs[0].text)


class HFBackend:
    """transformers generation fallback (CPU or GPU). No guided decoding, so we
    rely on the prompt + schema validation to enforce structure."""

    def __init__(self, model: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def extract(self, image_bytes: bytes) -> dict:
        img = _pil(image_bytes)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": EXTRACT_PROMPT},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return _validate(raw)


def get_backend() -> Backend:
    name = os.environ.get("INVOICE_BACKEND", "vllm").lower()
    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HFBackend()
    if name == "vllm":
        quant = os.environ.get("INVOICE_QUANTIZATION") or None
        return VLLMBackend(quantization=quant)
    raise ValueError(f"unknown INVOICE_BACKEND: {name!r}")
