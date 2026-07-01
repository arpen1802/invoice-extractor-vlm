# Implementation Plan — Invoice Extractor

Expands `COWORK_BRIEF_invoice_extractor.md` into concrete, reviewable tasks. Work milestone by milestone; stop at each checkpoint. Nothing here fabricates results — every metric ends up in the README only via a re-runnable script.

## Guiding constraints (carried from the brief)
- Metrics-first: `eval/metrics.py` exists **before** any training run.
- Real held-out test set of actual invoices, never touched by training or tuning.
- No invented numbers. Unrun results stay as `TODO: run`.
- Pinned deps, small commits, `.gitignore` for weights/data, stop before spending money (baseline API, rented GPU).

## GPU decision points (flagged — decide when reached)
Compute is undecided, so the plan keeps everything runnable on CPU until a GPU is actually required. GPU is needed at exactly three points, each a checkpoint:
1. **M3 tiny-loop:** proving the QLoRA training loop runs. Can be done on a small rented GPU (or Colab) for minutes.
2. **M4 full train:** the real fine-tune. Needs a real GPU for hours — cost estimate required before starting.
3. **M5 serving:** vLLM quantized inference benchmark. Needs a GPU host; a CPU-fallback demo path is provided for a cheap live demo.
Everything else (scaffold, data prep, metrics, baseline plumbing, schema, Docker build, Gradio wiring) runs on CPU.

---

## Milestone 0 — Scaffold  *(CPU)*
**Goal:** empty but coherent repo that imports cleanly.

Tasks:
- Create the folder tree from the brief (`data/`, `training/`, `eval/`, `serving/`, `demo/`, `notebooks/`).
- `serving/schema.py`: Pydantic v2 models for the fixed schema (`Invoice`, `LineItem`). This is the single source of truth reused by generation, decoding, and eval. Include field validators for date format and numeric coercion.
- `requirements.txt`: pinned. Split into core vs. GPU-only (comment the GPU block: `transformers`, `peft`, `bitsandbytes`, `vllm`) so CPU work installs cleanly.
- `.gitignore`: `data/`, `*.pt`, `*.safetensors`, `outputs/`, `wandb/`, `.env`, model caches.
- `README.md` skeleton: pitch, architecture placeholder, **results table with all cells = `TODO: run`**, run instructions stubs.
- `data/README.md` stub documenting where data will come from.

**Checkpoint:** show the tree, `schema.py`, and confirm `python -c "import serving.schema"` works.

---

## Milestone 1 — Data  *(CPU)*
**Goal:** train/val/test splits with a real held-out test set, plus a synthetic generator.

Tasks:
- `data/prepare.py`: download/ingest CORD; map CORD fields → our schema; write normalized `(image, invoice.json)` pairs.
- Decide the **held-out test set**: a slice of *real* invoices (CORD real receipts, plus DocILE/FUNSD-style if accessible). Document that it is frozen and never used for training/tuning.
- `data/synth/generate.py`: render labeled synthetic invoices (reportlab or HTML→image) + `faker` for vendors/amounts/dates. Vary layout, currency, line-item count. Target ~1–3k examples.
- Split policy: train = synthetic + majority of CORD; val = small mixed; test = real only.
- Print split counts; show 3 sample `(image, JSON)` pairs.
- Record all counts in `data/README.md`.

Open questions to surface at checkpoint: is DocILE accessible? Enough real invoices to make the test set meaningful (aim ≥100)?

**Checkpoint:** split counts + 3 samples + confirmation the test set is real-only and frozen.

---

## Milestone 2 — Metrics + baseline plumbing  *(CPU; baseline run gated on cost)*
**Goal:** honest evaluation harness, ready before training.

Tasks:
- `eval/metrics.py`:
  - Per-field exact-match with normalization (`"1,488.00" == 1488.0`; format-tolerant dates; case/whitespace-tolerant strings).
  - Line-item F1 (bipartite match predicted↔ground-truth line items — the headline metric).
  - JSON validity rate (parses AND validates against `schema.py`).
  - Output as printable tables + a machine-readable JSON so the README table is populated by script, not by hand.
- Unit tests for normalization edge cases (thousands separators, currency symbols, `null` vs `0`, date formats).
- `eval/compare_baseline.py`: run a general API (GPT-4o / Claude) zero-shot on the **same held-out test set**, produce side-by-side table. **Build it but do not run it** — compute and report the estimated API cost (≈ test-set size × tokens/call) and wait for go-ahead.

**Checkpoint:** metrics pass unit tests; baseline script ready; API cost estimate presented for approval.

---

## Milestone 3 — Training harness  *(GPU decision point #1 — tiny loop)*
**Goal:** prove the QLoRA loop end-to-end on a tiny subset before spending on a full run.

Tasks:
- `training/config.yaml`: base model `Qwen2.5-VL-3B-Instruct`, QLoRA (4-bit) params, LoRA rank/alpha/targets, seq/image settings, batch/accumulation, LR schedule, W&B project.
- `training/train_qlora.py`: load base + bitsandbytes 4-bit, attach PEFT LoRA, format `(image, prompt → JSON)` examples, train, log to W&B, save adapter.
- Prove the loop on ~20 examples for a few steps (tiny GPU / Colab). Confirm loss logs, adapter saves, and a sample generation parses.
- Note GPU requirements (VRAM for 3B QLoRA + vision) and the fallback (text model + OCR) only if compute is a hard blocker — flag if invoked.

**Checkpoint:** tiny run logs in W&B + one parsed sample generation. Get go-ahead + confirmed GPU/cost before the full run.

---

## Milestone 4 — Full train + evaluate  *(GPU decision point #2 — full run)*
**Goal:** the real fine-tune and the honest results table.

Tasks:
- Run the full QLoRA fine-tune (help set up the rented GPU host if needed).
- Add guided/constrained JSON decoding (Outlines / XGrammar / vLLM guided) driven by `schema.py`.
- Full eval on the frozen held-out set via `eval/metrics.py`; run the approved baseline comparison.
- **Fill the README results table with real numbers** (field acc, line-item F1, JSON validity, fine-tuned vs API). Report truthfully even where the API wins.
- Error analysis on the worst fields; propose one targeted fix (e.g., synthetic data for the weak field, prompt tweak, decoding constraint).

**Checkpoint:** populated results table + error analysis + proposed fix.

---

## Milestone 5 — Productionize  *(GPU decision point #3 — serving; CPU-fallback demo)*
**Goal:** deployable stack + polished README.

Tasks:
- `serving/app.py`: FastAPI + vLLM, guided decoding, Pydantic validation on the way out.
- Quantized serving (4-bit / AWQ); benchmark latency + memory vs full precision — real numbers via a benchmark script.
- `serving/Dockerfile`: GPU target; ensure `docker build` succeeds.
- `demo/app.py`: Gradio upload → JSON. Provide a CPU-fallback path for a cheap live demo.
- Final `README.md`: pitch, architecture diagram, results table, run instructions, demo GIF, design tradeoffs (why small fine-tuned VLM, quantization tradeoffs, synthetic-data caveats).

**Checkpoint:** `docker build` succeeds; endpoint returns schema-valid JSON; Gradio runs; README complete.

---

## Definition of done (from the brief)
- Every README metric reproducible from repo scripts on the real held-out set.
- Fine-tuned-vs-API table exists and is honest.
- `docker build` succeeds; FastAPI returns schema-valid JSON; Gradio demo runs.
- README explains design decisions and tradeoffs.

## Suggested order of first actions
1. M0 scaffold (fast, unblocks everything).
2. M1 data — the real bottleneck; the held-out set quality gates the whole project's credibility.
3. M2 metrics before any GPU spend.
Stop after each for review.
