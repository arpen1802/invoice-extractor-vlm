# Data

Nothing under `data/` is committed except this file and the generator scripts
(see `.gitignore`). Regenerate or re-fetch locally with the scripts below.

## Sources (planned — Milestone 1)
- **CORD** (receipts) — real documents, ingested and mapped to our schema via
  `data/prepare.py`. A real slice becomes part of the held-out test set.
- **DocILE / FUNSD-style invoices** — used if accessible, to strengthen the
  real-invoice test set. _TODO: confirm access._
- **Synthetic** — `data/synth/generate.py` renders labeled invoices (reportlab /
  HTML→image) with `faker` for vendors/amounts/dates. Varies layout, currency,
  and line-item count. Target ~1–3k training examples.

## Splits
- **train** = synthetic + majority of CORD
- **val** = small mixed set
- **test** = **real invoices only, frozen** — never used for training or any
  tuning decision. Headline metrics are reported on this set only.

## CORD → schema mapping (caveat)
CORD is Indonesian **receipts**, not invoices. Receipts reliably carry line items
and totals but usually lack `invoice_number`, `vendor_tax_id`, `due_date`, and a
normalized `invoice_date`. `prepare.py` maps what's present, sets the rest to
`null`, defaults `currency` to `IDR`, and **drops** any example missing a
required-non-null field (`invoice_number`, `invoice_date`, `vendor_name`,
`currency`, `total`) rather than padding with fabricated values. It reports how
many were dropped. This keeps the real test set honest even though it shrinks it.

## Counts
Synthetic generation + split plumbing verified (synthetic-only dry run: e.g. 40
synth → train 36 / val 4 / test 0). Real counts are filled once the CORD path is
run on a machine with Hugging Face access.
_TODO: run `data/prepare.py --with-cord` and record actual train / val / test
counts and how many real invoices land in test._

## How to regenerate
```bash
python data/synth/generate.py --n 1500 --seed 0   # synthetic (image, label) pairs
python data/prepare.py --with-cord --seed 0       # + CORD ingest, mapping, splits
# offline check (no HF): synthetic-only, empty test set
python data/prepare.py --no-cord --seed 0
```
Manifests land in `data/splits/{train,val,test}.jsonl`; each line is
`{"image": ..., "label": ..., "source": "synth"|"cord"}` with repo-relative paths.
