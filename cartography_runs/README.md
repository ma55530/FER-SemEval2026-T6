# Cartography Runs

## Included

- `scripts/semeval_cartography.py`: full multitask cartography with constrained decoding, independent-head, and joint-pair variants. All three variants default to seeds `42,123,456,789,1337`.
- `scripts/semeval_clarity_cartography.py`: flat and hierarchical clarity cartography. Both variants default to seeds `42,123,456,789,1337` and export cross-seed stability files.
- `notebooks/full_cartography.ipynb`, `notebooks/flat_multitask.ipynb`, `notebooks/hierarchical_clarity.ipynb`: cleaned source notebooks with execution outputs removed.
- `Dockerfile`, `run_full_cartography.sh`, `run_clarity_cartography.sh`, and `extract_results.sh`: Docker/native launch and result packaging helpers.

## Docker

The Dockerfile defaults to CUDA 12.1/cu121 for broad NVIDIA GPU compatibility:

```bash
bash run_full_cartography.sh docker
bash run_clarity_cartography.sh docker
```

For Blackwell-class GPUs that require CUDA 12.8/cu128:

```bash
CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
bash run_full_cartography.sh docker
```

The same build args work for the clarity runner. Put Hugging Face tokens in a local `.env` file when needed; it is ignored by git and Docker build context.

## Quick Checks

```bash
bash run_full_cartography.sh docker --dry-run
bash run_clarity_cartography.sh docker --dry-run
```

Use `extract_results.sh` after Docker runs to copy `/app/output` from the containers and create a timestamped archive.
