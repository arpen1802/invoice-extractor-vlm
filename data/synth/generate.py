"""Synthetic invoice generator.

Renders labeled invoice *images* (PNG) directly with Pillow — no PDF/poppler
dependency, so it runs anywhere including a Mac. For each invoice we emit:

    data/synth/images/<id>.png
    data/synth/labels/<id>.json      # validated against serving.schema.Invoice

Variety is the point: multiple layout templates, fonts, currencies, tax regimes,
and line-item counts, so a fine-tune doesn't overfit one look. Ground-truth JSON
is generated *first*, then rendered, so labels are exact by construction.

Usage:
    python data/synth/generate.py --n 1000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Import the shared schema (repo root on path).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serving.schema import Invoice  # noqa: E402

OUT = Path(__file__).resolve().parent
IMAGES = OUT / "images"
LABELS = OUT / "labels"

# Currency -> (symbol, typical tax rate range). Mix of regions.
CURRENCIES = {
    "USD": ("$", (0.0, 0.095)),
    "EUR": ("€", (0.17, 0.25)),
    "GBP": ("£", (0.0, 0.20)),
    "JPY": ("¥", (0.08, 0.10)),
    "CAD": ("$", (0.05, 0.15)),
    "AUD": ("$", (0.10, 0.10)),
}

# Currencies with no minor unit — amounts must be whole numbers so the rendered
# image (which prints no decimals) matches the label exactly.
ZERO_DECIMAL = {"JPY"}

FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/liberation2",
    "/Library/Fonts",  # macOS fallback
    "/System/Library/Fonts",
]

# (regular, bold) font-file basenames we try to find across the dirs above.
FONT_SETS = [
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
]

PRODUCTS = [
    "Consulting services", "Software license", "Cloud hosting", "Design work",
    "Widget assembly", "Maintenance plan", "Freight & shipping", "Installation",
    "Training session", "Support retainer", "Hardware unit", "Data migration",
    "Steel bracket", "Office supplies", "Marketing campaign", "Legal review",
]


def _find_font(basename: str):
    for d in FONT_DIRS:
        p = Path(d) / basename
        if p.exists():
            return str(p)
    return None


def _load_font_set(rng: random.Random):
    """Return (regular_path, bold_path), falling back to any available pair."""
    order = rng.sample(FONT_SETS, len(FONT_SETS))
    for reg, bold in order:
        rp, bp = _find_font(reg), _find_font(bold)
        if rp and bp:
            return rp, bp
    # Last resort: PIL default (bitmap) — rendering still works.
    return None, None


def _fmt_money(value: float, currency: str) -> str:
    """Human-rendered amount for the *image* (may include symbol/separators).

    The label JSON keeps the clean numeric value; the schema's normalizer will
    reconcile any string form during eval, so we can render prettily here.
    """
    sym, _ = CURRENCIES[currency]
    if currency == "JPY":
        return f"{sym}{value:,.0f}"
    return f"{sym}{value:,.2f}"


def _make_record(rng: random.Random, fake: Faker) -> dict:
    """Build ground-truth invoice data (as a plain dict) BEFORE rendering."""
    currency = rng.choice(list(CURRENCIES))
    _, (tax_lo, tax_hi) = CURRENCIES[currency]
    tax_rate = round(rng.uniform(tax_lo, tax_hi), 4)

    zero_dec = currency in ZERO_DECIMAL
    rnd = (lambda v: round(v)) if zero_dec else (lambda v: round(v, 2))

    n_items = rng.randint(1, 7)
    line_items = []
    subtotal = 0.0
    for _ in range(n_items):
        qty = rng.randint(1, 12)
        unit = rnd(rng.uniform(5, 2000))
        amount = rnd(qty * unit)
        subtotal += amount
        line_items.append({
            "description": rng.choice(PRODUCTS),
            "quantity": float(qty),
            "unit_price": unit,
            "amount": amount,
        })
    subtotal = rnd(subtotal)
    tax_amount = rnd(subtotal * tax_rate)
    total = rnd(subtotal + tax_amount)

    inv_date = fake.date_between(start_date="-2y", end_date="today")
    has_due = rng.random() < 0.8
    due_date = fake.date_between(start_date=inv_date, end_date="+60d") if has_due else None

    record = {
        "invoice_number": f"INV-{inv_date.year}-{rng.randint(1000, 9999)}",
        "invoice_date": inv_date.isoformat(),
        "due_date": due_date.isoformat() if due_date else None,
        "vendor_name": fake.company(),
        "vendor_tax_id": fake.bothify("??######") if rng.random() < 0.6 else None,
        "bill_to": fake.company() if rng.random() < 0.9 else None,
        "currency": currency,
        "subtotal": subtotal,
        "tax_amount": tax_amount,
        "total": total,
        "line_items": line_items,
    }
    return record


# ---- Rendering -------------------------------------------------------------

def _draw_invoice(record: dict, rng: random.Random, reg_path, bold_path) -> Image.Image:
    """Render an invoice image from the ground-truth record.

    Two broad templates (left-aligned header vs. right-aligned totals block),
    with jittered spacing so no two look identical.
    """
    W, H = 1000, 1400
    bg = rng.choice([(255, 255, 255), (252, 252, 250), (250, 251, 253)])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    def font(size, bold=False):
        path = bold_path if bold else reg_path
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    ink = (20, 20, 24)
    muted = (90, 90, 100)
    accent = rng.choice([(30, 60, 120), (120, 40, 40), (30, 100, 70), (40, 40, 40)])
    cur = record["currency"]

    margin = rng.randint(50, 80)
    y = margin

    # Header: vendor + INVOICE title.
    d.text((margin, y), record["vendor_name"], font=font(34, bold=True), fill=accent)
    title = "INVOICE"
    tf = font(38, bold=True)
    tw = d.textlength(title, font=tf)
    d.text((W - margin - tw, y), title, font=tf, fill=ink)
    y += 60
    if record["vendor_tax_id"]:
        d.text((margin, y), f"Tax ID: {record['vendor_tax_id']}", font=font(16), fill=muted)
    y += 40

    # Meta block (number / dates).
    meta = [("Invoice #", record["invoice_number"]),
            ("Date", record["invoice_date"])]
    if record["due_date"]:
        meta.append(("Due", record["due_date"]))
    mx = W - margin - 300
    my = y
    for label, val in meta:
        d.text((mx, my), label, font=font(15), fill=muted)
        d.text((mx + 110, my), str(val), font=font(15, bold=True), fill=ink)
        my += 26

    if record["bill_to"]:
        d.text((margin, y), "Bill To", font=font(15), fill=muted)
        d.text((margin, y + 22), record["bill_to"], font=font(18, bold=True), fill=ink)
    y = max(y + 90, my + 20)

    # Line-item table.
    cols = [margin, margin + 430, margin + 560, margin + 720, W - margin]
    headers = ["Description", "Qty", "Unit", "Amount"]
    d.rectangle([margin, y, W - margin, y + 32], fill=accent)
    hy = y + 7
    d.text((cols[0] + 8, hy), headers[0], font=font(16, bold=True), fill=(255, 255, 255))
    for i, h in enumerate(headers[1:], start=1):
        d.text((cols[i], hy), h, font=font(16, bold=True), fill=(255, 255, 255))
    y += 40

    for it in record["line_items"]:
        d.text((cols[0] + 8, y), str(it["description"]), font=font(15), fill=ink)
        d.text((cols[1], y), f"{it['quantity']:.0f}", font=font(15), fill=ink)
        d.text((cols[2], y), _fmt_money(it["unit_price"], cur), font=font(15), fill=ink)
        d.text((cols[3], y), _fmt_money(it["amount"], cur), font=font(15), fill=ink)
        y += 30
        d.line([margin, y - 6, W - margin, y - 6], fill=(230, 230, 232))

    # Totals block (right-aligned).
    y += 20
    ty = y
    for label, val in [("Subtotal", record["subtotal"]),
                       ("Tax", record["tax_amount"]),
                       ("Total", record["total"])]:
        bold = label == "Total"
        d.text((cols[2], ty), label, font=font(17, bold=bold), fill=ink)
        amt = _fmt_money(val, cur)
        aw = d.textlength(amt, font=font(17, bold=bold))
        d.text((cols[4] - aw, ty), amt, font=font(17, bold=bold), fill=ink)
        ty += 30
    d.line([cols[2], y - 6, cols[4], y - 6], fill=ink)

    return img


def generate(n: int, seed: int) -> int:
    rng = random.Random(seed)
    fake = Faker()
    Faker.seed(seed)
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    written = 0
    for i in range(n):
        record = _make_record(rng, fake)
        # Validate label against the shared schema before saving.
        inv = Invoice(**record)
        clean = json.loads(inv.model_dump_json())

        reg_path, bold_path = _load_font_set(rng)
        img = _draw_invoice(record, rng, reg_path, bold_path)

        stem = f"synth_{i:06d}"
        img.save(IMAGES / f"{stem}.png")
        (LABELS / f"{stem}.json").write_text(json.dumps(clean, indent=2))
        written += 1

    print(f"Wrote {written} synthetic invoices to {IMAGES} and {LABELS}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic labeled invoices.")
    ap.add_argument("--n", type=int, default=1000, help="number of invoices")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.n, args.seed)


if __name__ == "__main__":
    main()
