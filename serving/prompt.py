"""Canonical extraction instruction.

Shared by training and serving so the model sees the *same* prompt at fine-tune
time and at inference. Keep this the single source of truth for the instruction;
if you change it, retrain (train/serve prompt drift silently hurts accuracy).
"""

from __future__ import annotations

EXTRACT_PROMPT = (
    "You are an invoice data extractor. Read the invoice image and return ONLY a "
    "single JSON object with exactly these keys:\n"
    "  invoice_number (string)\n"
    "  invoice_date (YYYY-MM-DD)\n"
    "  due_date (YYYY-MM-DD or null)\n"
    "  vendor_name (string)\n"
    "  vendor_tax_id (string or null)\n"
    "  bill_to (string or null)\n"
    "  currency (ISO 4217 code, e.g. USD)\n"
    "  subtotal (number or null)\n"
    "  tax_amount (number or null)\n"
    "  total (number)\n"
    "  line_items (array of objects with keys: description, quantity, "
    "unit_price, amount; use null for missing numbers)\n"
    "Use null where a value is not present on the invoice. "
    "Output valid JSON only, with no markdown fences and no explanation."
)
