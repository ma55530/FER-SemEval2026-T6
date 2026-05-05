#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-docker}"
if [ "$#" -gt 0 ]; then
  shift
fi

IMAGE_NAME="${IMAGE_NAME:-cartography-runs}"
CONTAINER_NAME="${CONTAINER_NAME:-cartography-clarity}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
EXTRA_ARGS=("$@")

if [ "$MODE" = "docker" ]; then
  docker build \
    --build-arg CUDA_IMAGE="$CUDA_IMAGE" \
    --build-arg TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    -t "$IMAGE_NAME" .

  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  ENV_FLAG=()
  if [ -f ".env" ]; then
    ENV_FLAG=(--env-file .env)
  fi

  docker volume create cartography-output >/dev/null
  docker volume create cartography-hf-cache >/dev/null

  docker run --gpus all \
    -v cartography-output:/app/output \
    -v cartography-hf-cache:/app/hf_cache \
    "${ENV_FLAG[@]}" \
    --restart on-failure:3 \
    --entrypoint python \
    --name "$CONTAINER_NAME" \
    -d \
    "$IMAGE_NAME" scripts/semeval_clarity_cartography.py "${EXTRA_ARGS[@]}"

  docker logs -f "$CONTAINER_NAME"
elif [ "$MODE" = "native" ]; then
  if [ ! -d "venv" ]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  python -m pip install --upgrade pip
  if [ -n "$TORCH_INDEX_URL" ]; then
    pip install torch --index-url "$TORCH_INDEX_URL"
  else
    pip install torch
  fi
  pip install -r requirements.txt
  mkdir -p output
  export OUTPUT_DIR=./output
  python scripts/semeval_clarity_cartography.py "${EXTRA_ARGS[@]}"
else
  echo "Usage: bash run_clarity_cartography.sh [docker|native] [script args...]"
  exit 1
fi
