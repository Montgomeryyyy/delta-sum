#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

MODE=${1}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

if [[ "$MODE" == "structured" ]]; then
    CONFIG_NAME="structured.yaml"
    DATASET_DIR="${STRUCTURED_PRE_TRAINING_DIR}/tokenized_dataset"
    TOKENIZER_DIR="${STRUCTURED_PRE_TRAINING_DIR}/tokenizer"
    OUTPUT_DIR="${STRUCTURED_MODEL_DIR}/pretraining"
else
    CONFIG_NAME="natural_language.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/tokenized_dataset"
    TOKENIZER_DIR="${HF_MODEL_NAME}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/pretraining"
fi

uv run pre_training_train \
    --config-name="${CONFIG_NAME}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    model.name="${HF_MODEL_NAME}" \
    tokenizer_dir="${TOKENIZER_DIR}"
