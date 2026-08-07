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

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" && "$MODE" != "natural_language_lora" ]]; then
    echo "Usage: $0 [structured|natural_language|natural_language_lora] [MODEL_DIR] [OUTPUT_DIRECTORY_NAME] [LORA_ADAPTER_PATH (if natural_language_lora)]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

MODEL_DIR=${2}
OUTPUT_DIRECTORY_NAME=${3}
LORA_ADAPTER_PATH=${4:-null}
if [[ "$MODE" != "natural_language_lora" ]]; then
    LORA_ADAPTER_PATH="null"
fi

# --- Pipeline Execution ---

# Print commands
# set -x

if [[ "$MODE" == "structured" ]]; then
    TASK_DATA_DIR="${STRUCTURED_TASK_DATA_DIR}"
    EMBEDDING_OUTPUT_DIR="${STRUCTURED_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/task_embeddings"
else
    TASK_DATA_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
    EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/task_embeddings"
fi

# --- Create Embeddings ---

for TASK_NAME in $ACES_TASKS; do
    echo "Processing task: ${TASK_NAME}"

    uv run embedding_and_lr_inference_embedding \
        model.pretrained_model_name_or_path="${MODEL_DIR}" \
        lora_adapter_path="${LORA_ADAPTER_PATH}" \
        dataset_dir="${TASK_DATA_DIR}/tokenized" \
        output_dir="${EMBEDDING_OUTPUT_DIR}" \
        num_workers="${NUM_WORKERS}" \
        task_name="${TASK_NAME}"

    # Aggressively delete HF_DATASETS_CACHE to prevent disk space exhaustion
    echo "Deleting ${HF_DATASETS_CACHE}"
    rm -rf "${HF_DATASETS_CACHE}"
done
