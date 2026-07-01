"""Baseline: run a general-purpose LLM API zero-shot on the SAME held-out test
set, so we can put fine-tuned-vs-API side by side in the README.

SPENDING GUARD
--------------
This script defaults to **--dry-run**: it estimates token usage and dollar cost
from the test manifest and prints it, WITHOUT calling any API. It only makes paid
calls when you pass **--run** *and* set the relevant API key. This matches the
brief's rule: stop before spending money and report the estimate first.

Usage
-----
    # cost estimate only (no network, no spend):
    python eval/compare_baseline.py --ref data/splits/test.jsonl --provider openai

    # actually run (costs money — needs OPENAI_API_KEY or ANTHROPIC_API_KEY):
    python eval/compare_baseline.py --ref data/splits/test.jsonl --provider openai --run \
        --out eval/preds_baseline.jsonl

Then score with:
    python eval/metrics.py --pred eval/preds_baseline.jsonl --ref data/splits/test.jsonl \
        --manifest --title "Baseline API"
"""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.schema import Invoice  # noqa: E402

# ---- Pricing (USD per 1M tokens). VERIFY before trusting the estimate; provider
# prices change. These are placeholders you can update in one place. ----------
@dataclass
class Pricing:
    name: str
    in_per_m: float          # $ / 1M input tokens
    out_per_m: float         # $ / 1M output tokens
    img_tokens: int          # approx input tokens charged per invoice image

PRICING = {
    # NOTE: confirm current numbers at the provider's pricing page before relying on these.
    "openai": Pricing("gpt-4o", in_per_m=2.50, out_per_m=10.00, img_tokens=1000),
    "anthropic": Pricing("claude-3.5-sonnet", in_per_m=3.00, out_per_m=15.00, img_tokens=1200),
}

PROMPT = (
    "You are an invoice data extractor. Read the invoice image and return ONLY a "
    "JSON object with exactly these keys: invoice_number, invoice_date (YYYY-MM-DD), "
    "due_date (YYYY-MM-DD or null), vendor_name, vendor_tax_id (or null), bill_to "
    "(or null), currency (ISO 4217), subtotal (or null), tax_amount (or null), "
    "total (number), line_items (array of {description, quantity, unit_price, "
    "amount}). Use null where a value is not present. Output JSON only, no prose."
)
# Rough token accounting for the estimate.
PROMPT_TOKENS = 180
OUTPUT_TOKENS_EST = 350  # typical schema JSON for a multi-line invoice


def _load_manifest(ref: Path) -> list[dict]:
    return [json.loads(l) for l in ref.read_text().splitlines() if l.strip()]


def estimate_cost(n: int, provider: str) -> dict:
    p = PRICING[provider]
    in_tokens = n * (p.img_tokens + PROMPT_TOKENS)
    out_tokens = n * OUTPUT_TOKENS_EST
    cost = in_tokens / 1e6 * p.in_per_m + out_tokens / 1e6 * p.out_per_m
    return {
        "provider": provider, "model": p.name, "n_images": n,
        "input_tokens": in_tokens, "output_tokens": out_tokens,
        "est_cost_usd": round(cost, 2),
        "per_image_usd": round(cost / n, 4) if n else 0.0,
    }


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _call_openai(img_path: Path) -> str:
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=PRICING["openai"].name,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(img_path)}"}},
        ]}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return r.choices[0].message.content


def _call_anthropic(img_path: Path) -> str:
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(img_path)}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def run(ref: Path, provider: str, out: Path):
    rows = _load_manifest(ref)
    if not rows:
        print("No test rows found. Run data/prepare.py --with-cord first.")
        return
    call = _call_openai if provider == "openai" else _call_anthropic
    preds = []
    for i, row in enumerate(rows):
        img = ROOT / row["image"]
        try:
            raw = call(img)
            # keep raw; metrics.py will parse+validate. Store as string.
            preds.append(raw)
        except Exception as e:
            print(f"  [{i}] API/parse error: {type(e).__name__}: {e}")
            preds.append("{}")  # invalid-ish -> counts against the baseline honestly
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(rows)}")
    out.write_text("\n".join(json.dumps(p) if not isinstance(p, str) else p for p in preds))
    print(f"Wrote {len(preds)} predictions to {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="test splits manifest (image/label rows)")
    ap.add_argument("--provider", choices=list(PRICING), default="openai")
    ap.add_argument("--run", action="store_true", help="actually call the API (COSTS MONEY)")
    ap.add_argument("--out", default="eval/preds_baseline.jsonl")
    args = ap.parse_args()

    rows = _load_manifest(Path(args.ref))
    est = estimate_cost(len(rows), args.provider)
    print("Cost estimate (verify provider pricing in PRICING before trusting):")
    for k, v in est.items():
        print(f"  {k:<15} {v}")

    if not args.run:
        print("\nDRY RUN — no API called, no spend. Re-run with --run to execute.")
        return

    print(f"\n--run set: calling {est['model']} on {est['n_images']} images "
          f"(~${est['est_cost_usd']}). Ctrl-C to abort.")
    run(Path(args.ref), args.provider, Path(args.out))


if __name__ == "__main__":
    main()
