"""Unit tests for eval/metrics.py — run with: python -m pytest eval/test_metrics.py

Covers the normalization edge cases the honesty of the whole eval depends on,
plus line-item F1 correctness including duplicates and invalid JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import (  # noqa: E402
    _fields_equal, _item_matches, _max_bipartite_matching, evaluate,
)


def _base_invoice(**over):
    d = {
        "invoice_number": "INV-1", "invoice_date": "2024-01-02", "due_date": None,
        "vendor_name": "Acme", "vendor_tax_id": None, "bill_to": None,
        "currency": "USD", "subtotal": 100.0, "tax_amount": 0.0, "total": 100.0,
        "line_items": [{"description": "Widget", "quantity": 1, "unit_price": 100, "amount": 100}],
    }
    d.update(over)
    return d


# ---- field normalization ---------------------------------------------------

def test_number_thousands_separator():
    assert _fields_equal("total", "1,488.00", 1488.0)
    assert _fields_equal("subtotal", "$1,488.00", 1488)

def test_number_eu_style():
    assert _fields_equal("total", "1.234,56", 1234.56)

def test_money_tolerance():
    assert _fields_equal("total", 1488.001, 1488.0)
    assert not _fields_equal("total", 1488.5, 1488.0)

def test_null_vs_zero_distinct():
    # null and 0 are different values and must NOT be counted equal.
    assert not _fields_equal("tax_amount", None, 0)
    assert _fields_equal("tax_amount", None, None)

def test_date_formats():
    assert _fields_equal("invoice_date", "March 3, 2024", "2024-03-03")
    assert _fields_equal("invoice_date", "03/03/2024", "2024-03-03")  # dd/mm
    assert not _fields_equal("invoice_date", "2024-03-04", "2024-03-03")

def test_string_case_whitespace():
    assert _fields_equal("vendor_name", "  ACME   Corp ", "acme corp")
    assert not _fields_equal("vendor_name", "Acme Inc", "Acme Corp")

def test_currency_case_insensitive():
    assert _fields_equal("currency", "usd", "USD")


# ---- line-item matching ----------------------------------------------------

def test_item_match_anchor_amount():
    a = {"description": "Widget", "amount": "1,000.00"}
    b = {"description": "widget", "amount": 1000.0}
    assert _item_matches(a, b)

def test_item_mismatch_amount():
    a = {"description": "Widget", "amount": 999}
    b = {"description": "Widget", "amount": 1000}
    assert not _item_matches(a, b)

def test_bipartite_handles_duplicates():
    pred = [{"description": "X", "amount": 10}, {"description": "X", "amount": 10}]
    gt = [{"description": "X", "amount": 10}, {"description": "X", "amount": 10}]
    assert _max_bipartite_matching(pred, gt) == 2
    # one extra predicted duplicate should not create a phantom 3rd match
    pred3 = pred + [{"description": "X", "amount": 10}]
    assert _max_bipartite_matching(pred3, gt) == 2


# ---- end-to-end report -----------------------------------------------------

def test_perfect_prediction():
    ref = _base_invoice()
    rep = evaluate([dict(ref)], [ref])
    d = rep.as_dict()
    assert d["json_validity_rate"] == 1.0
    assert d["mean_field_accuracy"] == 1.0
    assert d["line_item"]["f1"] == 1.0

def test_invalid_json_counts_all_wrong():
    ref = _base_invoice()
    rep = evaluate(["{ not valid json", ], [ref])
    d = rep.as_dict()
    assert d["json_validity_rate"] == 0.0
    assert d["mean_field_accuracy"] == 0.0
    # gt line items still count against recall -> F1 == 0
    assert d["line_item"]["f1"] == 0.0

def test_partial_credit():
    ref = _base_invoice(total=100.0, vendor_name="Acme")
    pred = _base_invoice(total=999.0, vendor_name="Acme")  # wrong total, right vendor
    rep = evaluate([pred], [ref])
    accs = rep.field_accuracy()
    assert accs["vendor_name"] == 1.0
    assert accs["total"] == 0.0


if __name__ == "__main__":
    # Allow running without pytest.
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
