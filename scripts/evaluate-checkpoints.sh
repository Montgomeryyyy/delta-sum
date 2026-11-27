#!/usr/bin/env bash

# Usage:
#   source .env && source scripts/config.sh
#   mkdir -p "$SCRATCH_DIR/../logs/evaluate-checkpoints"
#   RUN=/path/to/pretraining/.../20261118_131225
#   sbatch \
#     --job-name=evaluate-checkpoints \
#     --partition=slam_gpu \
#     --constraint=l40s \
#     --time=2-00:00:00 \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --cpus-per-task=18 \
#     --mem=64G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/evaluate-checkpoints/%x-%j.out" \
#     scripts/evaluate-checkpoints.sh natural_language_summary \
#       train_run_dir="$RUN" \
#       tokenizer_dir=google/medgemma-1.5-4b-it \
#       tensorboard_metrics="[auroc,auprc]" \
#       poll_seconds=600
#
# Arguments:
#   CONFIG_NAME: name under models/config/evaluate_checkpoints/ (no .yaml)
#   Remaining args: Hydra overrides

set -euo pipefail

CONFIG_NAME=${1:-}
shift || true

if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name> [hydra overrides...]" >&2
    echo "Configs: natural_language | natural_language_summary" >&2
    exit 1
fi

source "./scripts/config.sh"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

if [[ "$CONFIG_NAME" == structured* ]]; then
    TASK_DATASET_DIR="${STRUCTURED_TASK_DATA_DIR}"
else
    TASK_DATASET_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
fi

module load cuda/12.2.1-gcc-13.2.0 2>/dev/null || true

echo "Evaluating checkpoints with config=${CONFIG_NAME}"
echo "task_dataset_dir=${TASK_DATASET_DIR}"

uv run pre_training_evaluate_checkpoints \
    --config-name="evaluate_checkpoints/${CONFIG_NAME}" \
    task_dataset_dir="${TASK_DATASET_DIR}" \
    "$@"
