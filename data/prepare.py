"""Build train / val / test splits for the invoice extractor.

Sources
-------
- **Synthetic** (data/synth): rendered by data/synth/generate.py. Training only.
- **CORD** (naver-clova-ix/cord-v2): real receipts, mapped best-effort into our
  invoice schema. CORD's own *test* split becomes our **real, frozen test set** —
  the only set headline metrics are reported on. CORD's *train* split is folded
  into our training data.

Split policy (from the brief)
-----------------------------
    train = synthetic + CORD-train
    val   = small mixed slice (held out from train)
    test  = CORD-test  (REAL ONLY, never used for training/tuning)

Output
------
For every example we materialize:
    data/real/images/<id>.png , data/real/labels/<id>.json   (CORD, mapped)
    (synthetic already lives in data/synth/)
and manifests:
    data/splits/train.jsonl , val.jsonl , test.jsonl
each line: {"image": <path>, "label": <path>, "source": "synth"|"cord"}

CORD is receipts, not invoices — an honest caveat. Receipts reliably carry
line items and totals but often lack invoice_number / vendor_tax_id / due_date.
We map what's present and set the rest to null. Examples missing a *required*
field (total, currency, vendor_name, invoice_number) are dropped and counted, so
the test set stays clean rather than padded with fabricated values.

Usage
-----
    python data/prepare.py --with-cord --val-frac 0.1 --seed 0
    python data/prepare.py --no-cord           # synthetic-only dry run (no test set)

Requires network + `datasets` for the CORD path (run on a machine with HF access).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.schema import Invoice  # noqa: E402

DATA = ROOT / "data"
SYNTH_IMAGES = DATA / "synth" / "images"
SYNTH_LABELS = DATA / "synth" / "labels"
REAL_IMAGES = DATA / "real" / "images"
REAL_LABELS = DATA / "real" / "labels"
SPLITS = DATA / "splits"

REQUIRED_NONNULL = ("invoice_number", "invoice_date", "vendor_name", "currency", "total")


def _rel(p) -> str:
    """Repo-relative path string, so manifests are portable across machines."""
    return str(Path(p).resolve().relative_to(ROOT))


# ---- CORD mapping ----------------------------------------------------------

def _num(x):
    """CORD prices arrive as strings like '45,132' or '10.000'. Let the schema
    normalizer do the heavy lifting; here we just pass through non-empty."""
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _map_cord_example(gt: dict) -> dict | None:
    """Map one CORD `gt_parse` dict into our schema dict, or None if unmappable.

    CORD structure (roughly):
        menu: [{nm, cnt, unitprice, price}, ...]  (may be a single dict)
        sub_total: {subtotal_price, tax_price, ...}
        total: {total_price, ...}
    Field names vary; we access defensively.
    """
    menu = gt.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]

    line_items = []
    for m in menu:
        if not isinstance(m, dict):
            continue
        line_items.append({
            "description": (m.get("nm") or "").strip() or "item",
            "quantity": _num(m.get("cnt")),
            "unit_price": _num(m.get("unitprice")),
            "amount": _num(m.get("price")),
        })

    sub = gt.get("sub_total", {}) or {}
    tot = gt.get("total", {}) or {}

    record = {
        # Receipts rarely carry a formal invoice number; use receipt id if any.
        "invoice_number": _num(tot.get("emoneyid") or gt.get("receipt_no")),
        "invoice_date": None,          # CORD gt has no reliable normalized date
        "due_date": None,
        "vendor_name": None,           # CORD gt has no reliable store-name field
        "vendor_tax_id": None,
        "bill_to": None,
        "currency": "IDR",             # CORD = Indonesian receipts
        "subtotal": _num(sub.get("subtotal_price")),
        "tax_amount": _num(sub.get("tax_price")),
        "total": _num(tot.get("total_price")),
        "line_items": line_items,
    }

    # Drop if a required-non-null field is missing (keeps test set honest).
    for f in REQUIRED_NONNULL:
        if record.get(f) in (None, ""):
            return None
    try:
        inv = Invoice(**record)
    except Exception:
        return None
    return json.loads(inv.model_dump_json())


def _load_cord():
    from datasets import load_dataset  # local import: only needed on HF path
    return load_dataset("naver-clova-ix/cord-v2")


def _materialize_cord(split_name: str, ds_split, out_prefix: str) -> list[dict]:
    """Save images+labels for a CORD split; return manifest entries."""
    REAL_IMAGES.mkdir(parents=True, exist_ok=True)
    REAL_LABELS.mkdir(parents=True, exist_ok=True)
    entries, dropped = [], 0
    for i, ex in enumerate(ds_split):
        gt = json.loads(ex["ground_truth"])["gt_parse"]
        mapped = _map_cord_example(gt)
        if mapped is None:
            dropped += 1
            continue
        stem = f"{out_prefix}_{i:06d}"
        img_path = REAL_IMAGES / f"{stem}.png"
        lbl_path = REAL_LABELS / f"{stem}.json"
        ex["image"].convert("RGB").save(img_path)
        lbl_path.write_text(json.dumps(mapped, indent=2))
        entries.append({"image": _rel(img_path), "label": _rel(lbl_path), "source": "cord"})
    print(f"  CORD {split_name}: kept {len(entries)}, dropped {dropped} (missing required fields)")
    return entries


# ---- Synthetic ------------------------------------------------------------

def _synth_entries() -> list[dict]:
    if not SYNTH_LABELS.exists():
        return []
    entries = []
    for lbl in sorted(SYNTH_LABELS.glob("*.json")):
        img = SYNTH_IMAGES / (lbl.stem + ".png")
        if img.exists():
            entries.append({"image": _rel(img), "label": _rel(lbl), "source": "synth"})
    return entries


# ---- Split assembly -------------------------------------------------------

def _write_manifest(name: str, entries: list[dict]):
    SPLITS.mkdir(parents=True, exist_ok=True)
    path = SPLITS / f"{name}.jsonl"
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def build(with_cord: bool, val_frac: float, seed: int):
    rng = random.Random(seed)

    synth = _synth_entries()
    rng.shuffle(synth)
    if not synth:
        print("WARNING: no synthetic data found. Run data/synth/generate.py first.")

    train, val, test = [], [], []

    # Val is carved from synthetic (small); train gets the rest of synthetic.
    n_val = int(len(synth) * val_frac)
    val += synth[:n_val]
    train += synth[n_val:]

    if with_cord:
        try:
            ds = _load_cord()
        except Exception as e:
            print(f"WARNING: could not load CORD ({type(e).__name__}: {e}).")
            print("         Proceeding synthetic-only; TEST SET WILL BE EMPTY.")
            ds = None
        if ds is not None:
            train += _materialize_cord("train", ds["train"], "cord_train")
            # A few real receipts into val for a mixed val set.
            val += _materialize_cord("val", ds["validation"], "cord_val")
            # Real held-out test set — the headline metric set.
            test += _materialize_cord("test", ds["test"], "cord_test")
    else:
        print("NOTE: --no-cord dry run; test set intentionally empty.")

    rng.shuffle(train)
    _write_manifest("train", train)
    _write_manifest("val", val)
    _write_manifest("test", test)

    print("\nSplit counts")
    print(f"  train: {len(train)}  (synth {sum(e['source']=='synth' for e in train)}, "
          f"cord {sum(e['source']=='cord' for e in train)})")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}  (REAL only, frozen)")
    if not test:
        print("  ! test is empty — real held-out set requires the CORD path (HF access).")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--with-cord", dest="with_cord", action="store_true", default=True)
    g.add_argument("--no-cord", dest="with_cord", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.with_cord, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
