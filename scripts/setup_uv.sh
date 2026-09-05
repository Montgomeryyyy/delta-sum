#!/bin/bash

# Exit immediately if any command fails
set -e

module load cuda/12.2.1-gcc-13.2.0

# Print GPU status at launch
echo "Running on ${HOSTNAME}"
nvidia-smi
nvcc --version

# set -x

# Set env vars
source "./scripts/config.sh"
export MAX_JOBS=$SLURM_NTASKS
export NVCC_THREADS=1

# Setup uv
# rm -rf $UV_PROJECT_ENVIRONMENT
uv sync
