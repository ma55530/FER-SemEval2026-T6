ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip && \
    if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir torch --index-url "$TORCH_INDEX_URL"; \
    else \
        pip install --no-cache-dir torch; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts ./scripts
COPY notebooks ./notebooks

RUN mkdir -p /app/output /app/hf_cache

ENV OUTPUT_DIR=/app/output
ENV HF_HOME=/app/hf_cache
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["python", "scripts/semeval_cartography.py"]
