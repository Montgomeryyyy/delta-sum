#!/usr/bin/env bash

# Usage:
#   # After create-task-datasets finishes (replace JOBID):
#   source .env && source scripts/config.sh
#   mkdir -p "$SCRATCH_DIR/../logs/evaluate-tasks"
#   CKPT=.../checkpoint-541
#   sbatch \
#     --dependency=afterok:JOBID \
#     --job-name=evaluate-tasks \
#     --partition=slam_gpu \
#     --constraint=l40s \
#     --time=8:00:00 \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --cpus-per-task=18 \
#     --mem=64G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/evaluate-tasks/%x-%j.out" \
#     scripts/evaluate-tasks.sh natural_language \
#       "models={summary_medgemma_ckpt541: \"$CKPT\"}" \
#       tokenizer_dir=google/gemma-3-270m
#
# Arguments:
#   CONFIG_NAME: Hydra config under models/config/evaluate_tasks/ (no .yaml)
#   Remaining args: Hydra overrides (models=..., tokenizer_dir=..., ...)

set -euo pipefail

CONFIG_NAME=${1:-}
shift || true

if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name> [hydra overrides...]" >&2
    exit 1
fi

source "./scripts/config.sh"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

if [[ "$CONFIG_NAME" == structured* ]]; then
    TASK_DATASET_DIR="${STRUCTURED_TASK_DATA_DIR}"
    OUTPUT_DIR="${STRUCTURED_LR_INFERENCE_DIR}"
else
    TASK_DATASET_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_LR_INFERENCE_DIR}"
fi

module load cuda/12.2.1-gcc-13.2.0 2>/dev/null || true

echo "Evaluating with config=${CONFIG_NAME}"
echo "task_dataset_dir=${TASK_DATASET_DIR}"
echo "output_dir=${OUTPUT_DIR}"

uv run pre_training_evaluate_tasks \
    --config-name="${CONFIG_NAME}" \
    task_dataset_dir="${TASK_DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    "$@"
