"""Serving benchmark: latency + peak GPU memory, full precision vs quantized.

Produces the numbers for the README "Serving benchmark" table. Runs the chosen
backend over a sample of invoice images and records:
  - latency percentiles (p50 / p95 / mean) per request
  - peak GPU memory (via torch.cuda.max_memory_allocated)

Compare configs by running twice with different env and passing --label:
    INVOICE_QUANTIZATION= uvicorn... no; use directly:
    python serving/benchmark.py --images data/splits/test.jsonl --n 50 \
        --label full   --out serving/bench_full.json
    INVOICE_QUANTIZATION=awq python serving/benchmark.py --images ... \
        --label awq4   --out serving/bench_awq.json

GPU only for real numbers. On CPU / mock backend it still runs and reports
latency (memory will read 0), which is useful for wiring checks but NOT a
meaningful benchmark — never put mock/CPU numbers in the README.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_images(spec: str, n: int) -> list[bytes]:
    p = Path(spec)
    paths: list[Path] = []
    if p.suffix == ".jsonl":
        for line in p.read_text().splitlines():
            if line.strip():
                paths.append(ROOT / json.loads(line)["image"])
    elif p.is_dir():
        paths = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    else:
        paths = [p]
    paths = paths[:n]
    return [q.read_bytes() for q in paths if q.exists()]


def _peak_gpu_mem_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:
        pass
    return 0.0


def run(images: list[bytes], label: str, warmup: int) -> dict:
    from serving.backends import get_backend
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    backend = get_backend()

    # Warmup (excluded from timing).
    for i in range(min(warmup, len(images))):
        backend.extract(images[i])

    latencies, valid = [], 0
    for img in images:
        t0 = time.perf_counter()
        try:
            out = backend.extract(img)
            valid += 1 if out else 0
        except Exception:
            pass
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    def pct(p):
        if not latencies:
            return 0.0
        k = min(len(latencies) - 1, int(round(p / 100 * (len(latencies) - 1))))
        return round(latencies[k] * 1000, 1)

    return {
        "label": label,
        "n": len(images),
        "valid_outputs": valid,
        "latency_ms": {
            "p50": pct(50), "p95": pct(95),
            "mean": round(statistics.mean(latencies) * 1000, 1) if latencies else 0.0,
        },
        "peak_gpu_mem_mb": _peak_gpu_mem_mb(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="jsonl manifest, image dir, or single image")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    imgs = _load_images(args.images, args.n)
    if not imgs:
        print("No images found.")
        return
    res = run(imgs, args.label, args.warmup)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
