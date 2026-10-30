#!/bin/bash

# Exit immediately if any command fails
set -e

# Optional CUDA module (ERC default). Override or clear on other clusters:
#   CUDA_MODULE=cuda/12.4 bash scripts/setup_uv.sh
#   CUDA_MODULE= bash scripts/setup_uv.sh
CUDA_MODULE="${CUDA_MODULE-cuda/12.2.1-gcc-13.2.0}"
if [ -n "$CUDA_MODULE" ]; then
    if module load "$CUDA_MODULE" 2>/dev/null; then
        echo "Loaded module: $CUDA_MODULE"
    else
        echo "Warning: could not load module '$CUDA_MODULE' (ok if CUDA is already on PATH)." >&2
    fi
fi

# Print GPU status at launch (best-effort)
echo "Running on ${HOSTNAME}"
nvidia-smi || echo "Warning: nvidia-smi not available." >&2
nvcc --version || echo "Warning: nvcc not available." >&2

# set -x

# Set env vars
source "./scripts/config.sh"
export MAX_JOBS="${SLURM_NTASKS:-1}"
export NVCC_THREADS=1

# Setup uv
# rm -rf $UV_PROJECT_ENVIRONMENT
uv sync
