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

# Print commands
# set -x

if [[ "$MODE" == "structured" ]]; then
    TASK_DATA_DIR="${STRUCTURED_TASK_DATA_DIR}"
    EMBEDDING_OUTPUT_DIR="${STRUCTURED_INFERENCE_DIR}/pt/task_embeddings"
else
    TASK_DATA_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
    if [[ -z "$2" ]]; then
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/zero_shot/task_embeddings"
    else
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/cpt/task_embeddings"
    fi
fi

# --- Create Embeddings ---

for TASK_NAME in $ACES_TASKS; do
    echo "Processing task: ${TASK_NAME}"

    uv run embedding_and_lr_inference_embedding \
        model.pretrained_model_name_or_path="${MODEL_DIR}" \
        dataset_dir="${TASK_DATA_DIR}" \
        output_dir="${EMBEDDING_OUTPUT_DIR}" \
        num_workers="${NUM_WORKERS}" \
        task_name="${TASK_NAME}"
done
