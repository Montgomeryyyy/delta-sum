#!/usr/bin/env bash

# Usage:
#   scripts/pretrain-summary-judge.sh \
#     input_dir=/path/to/summarized_dataset \
#     output_dir=/path/to/judged_output
#
#   # Named config (e.g. ARC judge):
#   scripts/pretrain-summary-judge.sh \
#     --config-name judge_delta_arc \
#     input_dir=/path/to/summarized_dataset \
#     output_dir=/path/to/judged_output
#
#   # Remote ARC API configs (model.backend=arc) do not need a GPU:
#   sbatch \
#     --job-name=judge-arc-lite \
#     --partition=slam_cpu \
#     --time=4:00:00 \
#     --ntasks=1 \
#     --cpus-per-task=8 \
#     --mem=32G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/summarize/%x-%j.out" \
#     scripts/pretrain-summary-judge.sh \
#       --config-name judge_delta_arc \
#       input_dir=/path/to/summarized_dataset \
#       output_dir=/path/to/judged_output
#
# Extra hydra overrides are passed through.

set -euo pipefail

source "./scripts/config.sh"

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# System nvcc is CUDA 11.5 and cannot compile FlashInfer's H100 (compute_90a) kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

"$UV_PROJECT_ENVIRONMENT/bin/python" \
    -m mimic_struct_predict.models.judge_summarize \
    "$@" \
    hydra.job.chdir=false
