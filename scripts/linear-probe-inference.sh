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
set -x

if [[ "$MODE" == "structured" ]]; then
    MODEL_DIR=${HF_MODEL_NAME}
    TASK_DATA_DIR=${STRUCTURED_TASK_DATA_DIR}
    OUTPUT_DIR=${STRUCTURED_INFERENCE_DIR}
    LR_MODEL_DIR=${STRUCTURED_MODEL_DIR}
else
    MODEL_DIR=${HF_MODEL_NAME}
    TASK_DATA_DIR=${NATURAL_LANGUAGE_TASK_DATA_DIR}
    OUTPUT_DIR=${NATURAL_LANGUAGE_INFERENCE_DIR}
    LR_MODEL_DIR=${NATURAL_LANGUAGE_MODEL_DIR}
fi

uv run embedding_and_lr_inference \
    model.pretrained_model_name_or_path="${MODEL_DIR}" \
    dataset_dir="${TASK_DATA_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    lr_model_dir="${LR_MODEL_DIR}/embedding_lr/models" \
    split=tuning
