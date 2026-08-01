#!/usr/bin/env bash

# Usage:
#   ./scripts/linear-probe-train.sh [MODE] [MODEL_DIR]
#
# Arguments:
#   MODE: Either 'natural_language' or 'structured'.
#   MODEL_DIR: Optional. Path to the model directory. Defaults to HF_MODEL_NAME.

# Exit immediately if a command exits with a non-zero status
set -e

MODE=${1}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language] [MODEL_DIR]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

MODEL_DIR=${2:-${HF_MODEL_NAME}}

# --- Pipeline Execution ---

IS_ZERO_SHOT=false
for arg in "$@"; do
    if [[ "$arg" == "--zero-shot" ]]; then
        IS_ZERO_SHOT=true
    fi
done

if [[ "$MODE" == "structured" ]]; then
    EMBEDDING_OUTPUT_DIR="${STRUCTURED_TASK_EMBEDDING_DIR}"
    LR_MODEL_DIR="${STRUCTURED_LR_MODEL_DIR}"
else
    if [[ "$IS_ZERO_SHOT" == true ]]; then
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/zero_shot/task_embeddings"
        LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/zero_shot/embedding_lr/models"
    else
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/cpt/task_embeddings"
        LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/cpt/embedding_lr/models"
    fi
fi

echo "--- Training Logistic Regression Models ---"
for task in $ACES_TASKS; do
    task_dir="${EMBEDDING_OUTPUT_DIR}/${task}"
    echo "Training embedding logistic regression for task: $task"
    uv run embedding_and_lr_train --multirun \
        task_dir="$task_dir" \
        output_dir="${LR_MODEL_DIR}/$task"
done
