#!/usr/bin/env bash

# Usage:
#   scripts/pretrain-summary-judge.sh \
#     input_dir=/path/to/summarized_dataset \
#     output_dir=/path/to/judged_output
#
# Extra hydra overrides are passed through. The prompt lives in
# mimic_struct_predict/models/config/judge/judge.yaml.

set -euo pipefail

source "./scripts/config.sh"

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# System nvcc is CUDA 11.5 and cannot compile FlashInfer's H100 (compute_90a) kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0

"$UV_PROJECT_ENVIRONMENT/bin/python" \
    -m mimic_struct_predict.models.judge_summarize \
    "$@" \
    hydra.job.chdir=false
