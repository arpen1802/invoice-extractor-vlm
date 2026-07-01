# Cowork Brief: Invoice Extractor (fine-tuned small VLM → structured JSON)

## How to use this
1. Create an empty project folder, e.g. `invoice-extractor/`, and open Cowork pointed at it.
2. Paste this whole file as the first instruction, or save it as `BRIEF.md` in the folder and tell Cowork: "Read BRIEF.md and start with Milestone 0, then stop for my review."
3. Work milestone by milestone. Do not let the agent do everything in one shot — review at each checkpoint.

---

## Role & goal
You are helping me build a CV/portfolio project: a **small vision-language model fine-tuned to extract a fixed invoice schema as strict JSON**, that beats a general-purpose LLM API on **field-level accuracy** for this narrow task, and is served as a production API with quantized inference.

The headline deliverable is an **honest accuracy comparison** (my fine-tuned 3B model vs. a general API, on real held-out invoices) plus a clean, deployable serving stack. Accuracy is the priority.

## Ground rules (important — do not violate)
- **Never fabricate or hard-code results, metrics, or numbers.** Every number in the README must come from a script in this repo that I can re-run. If something hasn't been run yet, write `TODO: run` — do not invent a plausible value.
- **Keep a real held-out test set of actual invoices** that is never used in training or for any tuning decision. Report headline metrics on this set only.
- **Write `eval/metrics.py` before training.** Metrics-first keeps the project honest.
- Prefer small, reviewable commits. Stop at each milestone checkpoint and summarize what you did and what you need from me.
- Pin dependency versions in `requirements.txt`. Note anything requiring a GPU so I can run it in the right environment.
- If you hit an API key or paid-resource need (the baseline API call, a rented GPU), stop and tell me exactly what's needed and the estimated cost — don't assume.
- Don't commit large model weights or datasets; use `.gitignore` and a `data/README.md` that documents how to fetch/generate data.

## Tech stack (use these unless you flag a good reason)
- **Base model:** Qwen2.5-VL-3B-Instruct (vision route — reads the invoice image directly, preserves layout). Fall back to a text model + OCR only if compute is a hard blocker; if so, flag it.
- **Fine-tuning:** QLoRA (4-bit) via Hugging Face `transformers` + `peft` + `bitsandbytes`.
- **Experiment tracking:** Weights & Biases.
- **Structured output:** guided/constrained JSON decoding (Outlines, XGrammar, or vLLM guided decoding).
- **Serving:** vLLM behind FastAPI. Pydantic schema validation.
- **Quantization at serve time:** 4-bit / AWQ; benchmark latency + memory vs. full precision.
- **Demo:** Gradio (upload invoice → JSON).
- **Container:** Docker. Target GPU host (RunPod/Vast/cloud); provide a CPU-fallback path for a cheap live demo if needed.

## The fixed schema (lock this on Day 1)
```json
{
  "invoice_number": "string",
  "invoice_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD | null",
  "vendor_name": "string",
  "vendor_tax_id": "string | null",
  "bill_to": "string | null",
  "currency": "ISO 4217 string",
  "subtotal": "number | null",
  "tax_amount": "number | null",
  "total": "number",
  "line_items": [
    {"description": "string", "quantity": "number | null",
     "unit_price": "number | null", "amount": "number | null"}
  ]
}
```
Implement as Pydantic models in `serving/schema.py` and reuse everywhere (generation, decoding, eval).

## Data plan
- **Real:** use the CORD receipt dataset (and, if accessible, DocILE/FUNSD-style invoice data) mapped into the schema above. A slice of real invoices is the **held-out test set**.
- **Synthetic:** write `data/synth/generate.py` to render labeled invoices (a templating/HTML-to-image or reportlab approach + `faker` for vendors/amounts/dates). Vary layouts, currencies, line-item counts. Target ~1–3k training examples.
- **Splits:** train = synthetic + majority of CORD; val = small mixed; **test = real invoices only, held out**. Document counts in `data/README.md`.

## Evaluation (this is the CV headline — get it right)
Implement in `eval/metrics.py`, reported as tables:
- **Per-field exact-match accuracy** (with numeric/date normalization: "1,488.00" == 1488.0; format-tolerant dates).
- **Line-item F1** (match predicted line items to ground truth — the hard, impressive metric).
- **JSON validity rate** (% parses and matches schema).
- **Baseline comparison:** `eval/compare_baseline.py` runs a general API (e.g. GPT-4o / Claude) zero-shot on the **same held-out test set**, and produces a side-by-side table vs. the fine-tuned model. This table is the centerpiece of the README.

## Repo structure to create
```
invoice-extractor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── synth/generate.py
│   └── prepare.py
├── training/
│   ├── train_qlora.py
│   └── config.yaml
├── eval/
│   ├── metrics.py
│   └── compare_baseline.py
├── serving/
│   ├── app.py          # FastAPI + vLLM
│   ├── schema.py       # Pydantic schema + guided decoding
│   └── Dockerfile
├── demo/app.py         # Gradio UI
└── notebooks/          # keep tidy
```

## Milestones (stop for my review after each)
**Milestone 0 — Scaffold.** Create the folder structure, `requirements.txt` (pinned), `.gitignore`, `serving/schema.py` (Pydantic), and a README skeleton with a results table full of `TODO: run` placeholders. Checkpoint.

**Milestone 1 — Data.** CORD ingestion + schema mapping, `data/synth/generate.py`, `data/prepare.py` producing train/val/test with a real held-out test set. Print split counts and show 3 sample (image, JSON) pairs. Checkpoint.

**Milestone 2 — Metrics + baseline.** Implement `eval/metrics.py` (field acc, line-item F1, JSON validity, normalization). Implement `eval/compare_baseline.py` but **stop before spending money** — tell me the API cost estimate and wait for my go-ahead to run it. Checkpoint.

**Milestone 3 — Training harness.** `train_qlora.py` + `config.yaml`, W&B logging, runnable end-to-end on a tiny subset first (prove the loop), GPU requirements noted. Checkpoint before a full run.

**Milestone 4 — Full train + evaluate.** Run/help me run the real QLoRA fine-tune, then full eval on the held-out set. Add guided JSON decoding. Fill the README results table with real numbers. Do error analysis on the worst fields and propose one targeted fix. Checkpoint.

**Milestone 5 — Productionize.** vLLM + FastAPI (`serving/app.py`), quantized serving with latency/memory benchmark, Dockerfile, Gradio demo, final README (pitch, architecture diagram, results table, run instructions, demo GIF). Checkpoint.

## Definition of done
- All metrics in the README are reproducible from repo scripts on the real held-out set.
- The fine-tuned-vs-API comparison table exists and is honest (even if the API wins some fields — report it truthfully).
- `docker build` succeeds; the FastAPI endpoint returns schema-valid JSON for an uploaded invoice; the Gradio demo runs.
- README explains design decisions and tradeoffs (why a small fine-tuned VLM, quantization tradeoffs, synthetic-data caveats).
