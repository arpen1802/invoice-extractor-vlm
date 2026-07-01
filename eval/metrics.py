"""Evaluation metrics for invoice extraction.

Three headline metrics, all reported on the frozen real held-out test set:

1. **Per-field exact-match accuracy** — with normalization so surface differences
   don't count as errors: "1,488.00" == 1488.0, format-tolerant dates,
   case/whitespace-tolerant strings, currency case-insensitive.
2. **Line-item F1** — predicted line items matched to ground truth via maximum
   bipartite matching (a pair matches iff normalized description AND amount
   agree). Micro-averaged across the set.
3. **JSON validity rate** — fraction of predictions that parse AND validate
   against serving.schema.Invoice.

Nothing here invents numbers: it consumes model predictions + ground-truth
labels and emits a report (printable tables + machine-readable dict) that the
README table is populated from.

Design notes
------------
- If a prediction fails to parse/validate, every field for that example counts
  wrong and its line items count as zero matches (predicted count still counts
  against precision only if we can read items; unparseable => 0 predicted).
- A field where BOTH prediction and reference are null counts as correct.
- Reusing serving.schema for normalization keeps generation/decoding/eval aligned.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.schema import Invoice, _coerce_date, _coerce_number  # noqa: E402

SCALAR_FIELDS = [
    "invoice_number", "invoice_date", "due_date", "vendor_name",
    "vendor_tax_id", "bill_to", "currency", "subtotal", "tax_amount", "total",
]
NUMERIC_FIELDS = {"subtotal", "tax_amount", "total"}
DATE_FIELDS = {"invoice_date", "due_date"}
STRING_FIELDS = {"invoice_number", "vendor_name", "vendor_tax_id", "bill_to"}

# Money compared to the cent; avoids float noise like 1488.0000001.
MONEY_TOL = 0.01


# ---- Normalization ---------------------------------------------------------

def _norm_string(v) -> Optional[str]:
    if v is None:
        return None
    s = re.sub(r"\s+", " ", str(v)).strip().casefold()
    return s or None


def _norm_number(v) -> Optional[float]:
    try:
        n = _coerce_number(v)
    except ValueError:
        return None
    return None if n is None else round(n, 2)


def _norm_date(v) -> Optional[str]:
    try:
        return _coerce_date(v)
    except ValueError:
        return None


def _norm_field(name: str, v):
    if name in NUMERIC_FIELDS:
        return _norm_number(v)
    if name in DATE_FIELDS:
        return _norm_date(v)
    if name == "currency":
        return None if v is None else str(v).strip().upper() or None
    return _norm_string(v)


def _fields_equal(name: str, a, b) -> bool:
    na, nb = _norm_field(name, a), _norm_field(name, b)
    if name in NUMERIC_FIELDS:
        if na is None or nb is None:
            return na is None and nb is None
        return abs(na - nb) <= MONEY_TOL
    return na == nb


# ---- Line-item matching ----------------------------------------------------

def _item_matches(a: dict, b: dict) -> bool:
    """A predicted item matches a ground-truth item iff normalized description
    and amount agree. Amount is the anchor; None==None allowed."""
    if _norm_string(a.get("description")) != _norm_string(b.get("description")):
        return False
    na, nb = _norm_number(a.get("amount")), _norm_number(b.get("amount"))
    if na is None or nb is None:
        return na is None and nb is None
    return abs(na - nb) <= MONEY_TOL


def _max_bipartite_matching(pred: list[dict], gt: list[dict]) -> int:
    """Maximum number of 1:1 matches between predicted and gt items.

    Augmenting-path (Kuhn's) algorithm over the equality graph — correct even
    when descriptions/amounts repeat across line items.
    """
    adj = [[j for j, g in enumerate(gt) if _item_matches(p, g)] for p in pred]
    match_gt = [-1] * len(gt)  # gt index -> pred index

    def try_assign(u, seen):
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                if match_gt[v] == -1 or try_assign(match_gt[v], seen):
                    match_gt[v] = u
                    return True
        return False

    matched = 0
    for u in range(len(pred)):
        if try_assign(u, [False] * len(gt)):
            matched += 1
    return matched


# ---- Report accumulation ---------------------------------------------------

@dataclass
class Report:
    n: int = 0
    valid_json: int = 0
    field_correct: dict = field(default_factory=lambda: {f: 0 for f in SCALAR_FIELDS})
    li_matched: int = 0
    li_pred: int = 0
    li_gt: int = 0

    def field_accuracy(self) -> dict:
        return {f: (self.field_correct[f] / self.n if self.n else 0.0) for f in SCALAR_FIELDS}

    def mean_field_accuracy(self) -> float:
        accs = self.field_accuracy().values()
        return sum(accs) / len(SCALAR_FIELDS) if self.n else 0.0

    def json_validity(self) -> float:
        return self.valid_json / self.n if self.n else 0.0

    def line_item_prf(self) -> tuple[float, float, float]:
        prec = self.li_matched / self.li_pred if self.li_pred else 0.0
        rec = self.li_matched / self.li_gt if self.li_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    def as_dict(self) -> dict:
        p, r, f1 = self.line_item_prf()
        return {
            "n": self.n,
            "json_validity_rate": round(self.json_validity(), 4),
            "field_accuracy": {k: round(v, 4) for k, v in self.field_accuracy().items()},
            "mean_field_accuracy": round(self.mean_field_accuracy(), 4),
            "line_item": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)},
        }


def _as_invoice_dict(pred) -> Optional[dict]:
    """Parse+validate a prediction (raw JSON string or dict) via the schema.
    Returns normalized dict, or None if invalid."""
    try:
        if isinstance(pred, str):
            pred = json.loads(pred)
        return json.loads(Invoice(**pred).model_dump_json())
    except Exception:
        return None


def evaluate(predictions: list, references: list[dict]) -> Report:
    """predictions: list of raw JSON strings or dicts (model outputs).
       references: list of ground-truth dicts (already schema-valid)."""
    assert len(predictions) == len(references), "pred/ref length mismatch"
    rep = Report()
    for pred_raw, ref in zip(predictions, references):
        rep.n += 1
        pred = _as_invoice_dict(pred_raw)

        if pred is None:
            # Invalid JSON: everything wrong; count gt line items against recall.
            rep.li_gt += len(ref.get("line_items", []))
            continue
        rep.valid_json += 1

        for f in SCALAR_FIELDS:
            if _fields_equal(f, pred.get(f), ref.get(f)):
                rep.field_correct[f] += 1

        p_items = pred.get("line_items", []) or []
        g_items = ref.get("line_items", []) or []
        rep.li_pred += len(p_items)
        rep.li_gt += len(g_items)
        rep.li_matched += _max_bipartite_matching(p_items, g_items)

    return rep


# ---- Pretty printing -------------------------------------------------------

def format_report(rep: Report, title: str = "Model") -> str:
    accs = rep.field_accuracy()
    p, r, f1 = rep.line_item_prf()
    lines = [f"== {title}  (n={rep.n}) ==", "", "Per-field exact-match accuracy:"]
    for f in SCALAR_FIELDS:
        lines.append(f"  {f:<16} {accs[f]*100:6.2f}%")
    lines += [
        "",
        f"Mean field accuracy : {rep.mean_field_accuracy()*100:6.2f}%",
        f"JSON validity rate  : {rep.json_validity()*100:6.2f}%",
        f"Line-item P/R/F1    : {p*100:5.1f}% / {r*100:5.1f}% / {f1*100:5.1f}%",
    ]
    return "\n".join(lines)


# ---- I/O helpers + CLI -----------------------------------------------------

def _load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _refs_from_manifest(manifest: Path) -> list[dict]:
    """Load ground-truth label dicts from a splits manifest (image/label rows)."""
    refs = []
    for row in _load_jsonl(manifest):
        refs.append(json.loads((ROOT / row["label"]).read_text()))
    return refs


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Score invoice predictions.")
    ap.add_argument("--pred", required=True, help="JSONL of model predictions (dict or raw-JSON-string per line)")
    ap.add_argument("--ref", required=True, help="ground-truth: a labels JSONL, or a splits manifest")
    ap.add_argument("--manifest", action="store_true", help="treat --ref as a splits manifest (image/label rows)")
    ap.add_argument("--out", help="write machine-readable report JSON here")
    ap.add_argument("--title", default="Model")
    args = ap.parse_args()

    preds = _load_jsonl(Path(args.pred))
    refs = _refs_from_manifest(Path(args.ref)) if args.manifest else _load_jsonl(Path(args.ref))

    rep = evaluate(preds, refs)
    print(format_report(rep, args.title))
    if args.out:
        Path(args.out).write_text(json.dumps(rep.as_dict(), indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
