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
invoice image ──► Qwen2.5-VL-3B (QLoRA fine-tuned) ──► guided JSON decoding ──► Pydantic validation ──► Invoice JSON
                                                        (schema in serving/schema.py)
```
_TODO: architecture diagram (Milestone 5)._

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
_TODO: fill per milestone (data prep, train, eval, serve, demo)._

## Compute note
Training (QLoRA 4-bit via bitsandbytes) requires an NVIDIA CUDA GPU and is done on
a rented/cloud GPU (RunPod/Vast). Data prep, metrics, the baseline API call, and
all serving/demo code run on CPU (including a MacBook). See `IMPLEMENTATION_PLAN.md`
for the three GPU decision points.

## Design decisions & tradeoffs
_TODO: expand (why small fine-tuned VLM, quantization tradeoffs, synthetic-data
caveats) — Milestone 5._
