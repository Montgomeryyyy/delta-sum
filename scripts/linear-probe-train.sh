#!/usr/bin/env bash

# Usage:
#   ./scripts/linear-probe-train.sh [MODE] []
#
# Arguments:
#   MODE: Either 'natural_language' or 'structured'.

# Exit immediately if a command exits with a non-zero status
set -e

MODE=${1}
OUTPUT_DIRECTORY_NAME=${2}

# --- Get config ---
source "./scripts/config.sh"


# --- Pipeline Execution ---

if [[ "$MODE" == "structured" ]]; then
    EMBEDDING_OUTPUT_DIR="${STRUCTURED_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/task_embeddings"
    LR_MODEL_DIR="${STRUCTURED_MODEL_DIR}/${OUTPUT_DIRECTORY_NAME}/embedding_lr/models"
elif [[ "$MODE" == "natural_language" ]] || [[ "$MODE" == "natural_language_lora" ]]; then
    EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/task_embeddings"
    LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/${OUTPUT_DIRECTORY_NAME}/embedding_lr/models"
else
    echo "Invalid MODE. Please choose one of: structured, natural_language, natural_language_lora"
    exit 1
fi

echo "--- Training Logistic Regression Models ---"
for task in $ACES_TASKS; do
    task_dir="${EMBEDDING_OUTPUT_DIR}/${task}"
    echo "Training embedding logistic regression for task: $task"
    uv run embedding_and_lr_train --multirun \
        task_dir="$task_dir" \
        output_dir="${LR_MODEL_DIR}/$task" \
        hydra/launcher=joblib \
        hydra.launcher.n_jobs=${NUM_WORKERS}
done
