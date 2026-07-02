# Invoice Extractor — fine-tuned small VLM → structured JSON

A small vision-language model (Qwen2.5-VL-3B-Instruct) fine-tuned with QLoRA to
extract a fixed invoice schema as **strict JSON**, aiming to beat a
general-purpose LLM API on **field-level accuracy** for this narrow task — served
as a production API with quantized inference.

> **Status:** scaffold (Milestone 0). All results below are placeholders until
> the corresponding scripts are run. No number appears here unless a script in
> this repo produced it.

## Why this project
General LLM APIs are strong but expensive and not layout-aware for documents. A
small VLM fine-tuned on one schema can be more accurate on that schema, cheaper
to run, and self-hostable. The headline is an **honest accuracy comparison** on a
real held-out test set, plus a clean deployable stack.

## Architecture
```
                 ┌─────────────────────────── training (offline, GPU) ──────────────────────────┐
  synthetic  ──► │ data/synth/generate.py ┐                                                       │
  CORD (real)──► │ data/prepare.py ────────┴─► train/val/test splits ─► train_qlora.py (QLoRA 4-bit)│
                 └───────────────────────────────────────────────┬──────────────────────────────┘
                                                                  ▼  LoRA adapter
  invoice image ──► FastAPI (serving/app.py) ──► Qwen2.5-VL-3B + adapter ──► guided JSON decoding
                                                 (vLLM, 4-bit/AWQ)           (schema-constrained)
                          │                                                          │
                          └──────────► Pydantic validation (serving/schema.py) ◄─────┘
                                                     ▼
                                            schema-valid Invoice JSON
  eval/metrics.py + eval/compare_baseline.py score ours vs a general API on the frozen real test set.
```

## Results (held-out real invoices)
All cells `TODO: run` until Milestone 4. Populated only by `eval/metrics.py` and
`eval/compare_baseline.py` on the frozen real-only test set.

### Per-field exact-match accuracy
| Field | Fine-tuned (ours) | Baseline API |
|---|---|---|
| invoice_number | TODO: run | TODO: run |
| invoice_date | TODO: run | TODO: run |
| due_date | TODO: run | TODO: run |
| vendor_name | TODO: run | TODO: run |
| vendor_tax_id | TODO: run | TODO: run |
| bill_to | TODO: run | TODO: run |
| currency | TODO: run | TODO: run |
| subtotal | TODO: run | TODO: run |
| tax_amount | TODO: run | TODO: run |
| total | TODO: run | TODO: run |

### Aggregate
| Metric | Fine-tuned (ours) | Baseline API |
|---|---|---|
| Line-item F1 | TODO: run | TODO: run |
| JSON validity rate | TODO: run | TODO: run |
| Mean field accuracy | TODO: run | TODO: run |

### Serving benchmark
| Config | Latency (p50/p95) | Peak GPU mem |
|---|---|---|
| Full precision | TODO: run | TODO: run |
| 4-bit / AWQ | TODO: run | TODO: run |

## The schema
Defined once in [`serving/schema.py`](serving/schema.py) (Pydantic) and reused for
generation, decoding, and evaluation. See that file for field-level details and
normalization rules.

## Repo layout
```
data/       CORD ingestion, synthetic generation, train/val/test prep
training/   QLoRA fine-tune (train_qlora.py + config.yaml)
eval/       metrics.py (field acc, line-item F1, JSON validity) + compare_baseline.py
serving/    FastAPI + vLLM app, schema.py, Dockerfile
demo/       Gradio upload → JSON
notebooks/  exploration
```

## Setup
```bash
pip install -r requirements.txt      # CPU work: data, metrics, baseline, serving code
# GPU host only (training/serving inference): see the GPU block in requirements.txt
```

## Run instructions
```bash
# 1. Data (CPU)
python data/synth/generate.py --n 1500 --seed 0
python data/prepare.py --with-cord --seed 0

# 2. Sanity-check the training pipeline without a GPU (CPU)
python training/train_qlora.py --config training/config.yaml --dry-run

# 3. Prove the training loop on a tiny subset, then full run (GPU host)
python training/train_qlora.py --config training/config.yaml --max-samples 20   # smoke
python training/train_qlora.py --config training/config.yaml                     # full

# 4. Evaluate + baseline (baseline dry-run first — see cost guard)
python eval/compare_baseline.py --ref data/splits/test.jsonl --provider openai   # estimate only
python eval/metrics.py --pred <preds>.jsonl --ref data/splits/test.jsonl --manifest
```
# 5. Serve (GPU) + demo
INVOICE_BACKEND=vllm INVOICE_MODEL=outputs/qlora-merged \
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
python serving/benchmark.py --images data/splits/test.jsonl --n 50 --label full --out serving/bench_full.json
INVOICE_QUANTIZATION=awq python serving/benchmark.py --images data/splits/test.jsonl --n 50 --label awq4 --out serving/bench_awq.json

# lightweight demo against the API, or fully local CPU fallback
INVOICE_API_URL=http://localhost:8000 python demo/app.py
INVOICE_DEMO_LOCAL=1 INVOICE_BACKEND=hf python demo/app.py   # CPU, slow, no vLLM

# API layer smoke test with no model / no GPU
INVOICE_BACKEND=mock uvicorn serving.app:app --port 8000
```

### Docker
```bash
docker build -f serving/Dockerfile -t invoice-extractor:latest .
docker run --gpus all -p 8000:8000 -e INVOICE_MODEL=/models/qlora-merged \
    -v $PWD/outputs:/models invoice-extractor:latest
```

### GPU requirements (training / inference)
QLoRA 4-bit of Qwen2.5-VL-3B + vision fits roughly **16–24 GB VRAM** at batch 1
with the image pixel caps in `training/config.yaml`. Training uses
`bitsandbytes` (CUDA only) — run it on a rented/cloud GPU (RunPod/Vast). The
`--dry-run` path is CPU-only and validates data/prompt formatting anywhere.

## Compute note
Training (QLoRA 4-bit via bitsandbytes) requires an NVIDIA CUDA GPU and is done on
a rented/cloud GPU (RunPod/Vast). Data prep, metrics, the baseline API call, and
all serving/demo code run on CPU (including a MacBook). See `IMPLEMENTATION_PLAN.md`
for the three GPU decision points.

## Design decisions & tradeoffs
**Why a small fine-tuned VLM instead of a general API.** For one fixed schema, a
3B model specialized on that schema can match or beat a large general model on
field accuracy while being cheap and self-hostable (no per-call fee, no data
leaving the host). The tradeoff is generality — this model only does invoices in
this schema, and a general API remains better for open-ended documents.

**Why the vision route (image in) over OCR + text model.** Reading the image
directly preserves layout — columns, alignment, and table structure carry
meaning for line items and totals that a flat OCR dump loses. The cost is more
GPU memory and vision-token budget, which we cap via image pixel limits in
`training/config.yaml`.

**Quantization tradeoffs.** 4-bit QLoRA makes fine-tuning fit on a modest GPU;
serving in 4-bit/AWQ cuts memory and can improve throughput, usually at a small
accuracy cost. The serving benchmark reports the full-vs-quantized numbers so the
tradeoff is measured, not assumed.

**Guided decoding.** Output is constrained to the JSON schema at decode time, so
malformed JSON is largely designed out rather than caught after the fact; Pydantic
validation is the final guard.

**Synthetic-data caveat.** Synthetic invoices give volume and perfect labels but
don't capture the full messiness of real documents (scans, noise, odd layouts).
That's exactly why headline metrics are reported only on a **real, frozen**
held-out set, never on synthetic data. Note also that the CORD source is
*receipts*, not invoices — see `data/README.md` for how that shapes the test set.

## Demo
_TODO: demo GIF (Milestone 5, after the model is trained)._
