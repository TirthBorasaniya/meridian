#!/bin/bash
# Fine-tune the domain-adapted embedder on one GPU. Pairs and negatives are built beforehand
# (`make embedder-pairs embedder-negatives`) and data/embedder/ is copied to the cluster checkout.
#SBATCH --job-name=meridian-embedder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_embedder_%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs .cache .tmp
export UV_CACHE_DIR="${PWD}/.cache/uv"
export HF_HOME="${PWD}/.cache/huggingface"
export TMPDIR="${PWD}/.tmp"
export PYTHONHASHSEED=0

uv sync --frozen --python 3.11.9
# pyproject pins Linux to CPU torch wheels for CI; the GPU job swaps in the CUDA build of the
# same version (set TORCH_CUDA_INDEX to match the cluster driver)
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
uv pip install --reinstall "torch==2.14.1" --index-url "${TORCH_CUDA_INDEX}"
nvidia-smi
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not active'"
uv run --no-sync meridian embedder train --config configs/embedder.yaml
