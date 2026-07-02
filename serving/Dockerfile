# Invoice Extractor serving image (GPU target).
#
# Build (from repo root):
#   docker build -f serving/Dockerfile -t invoice-extractor:latest .
# Run (needs NVIDIA Container Toolkit):
#   docker run --gpus all -p 8000:8000 \
#       -e INVOICE_BACKEND=vllm \
#       -e INVOICE_MODEL=/models/qlora-merged \
#       -v $PWD/outputs:/models invoice-extractor:latest
#
# CPU-only smoke of the API layer (no model): set INVOICE_BACKEND=mock.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    INVOICE_BACKEND=vllm

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Core deps first (cache layer), then GPU stack.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# GPU inference stack (kept out of requirements.txt so CPU installs stay light).
RUN pip install \
        torch==2.4.1 \
        transformers==4.45.2 \
        accelerate==1.0.1 \
        vllm==0.6.2 \
        qwen-vl-utils==0.0.8

COPY serving/ serving/
# schema/prompt live under serving/; ensure package import works.
ENV PYTHONPATH=/app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
