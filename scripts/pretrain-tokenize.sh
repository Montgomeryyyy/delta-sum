#!/usr/bin/env bash

# Usage:
#   ./scripts/pretrain-dataset.sh [MODE]
#
# Arguments:
#   MODE: Either 'natural_language' (default) or 'structured'.
#         Determines which configuration and output directory are used.

# Exit immediately if a command exits with a non-zero status
set -e

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
    DATASET_DIR="${STRUCTURED_MEDS_DIR}"
    OUTPUT_DIR="${STRUCTURED_PRE_TRAINING_DIR}"
    PRE_TRAINED_TOKENIZER=null
    TOKENIZER_ARG="${OUTPUT_DIR}/tokenizer"
else
    CONFIG_NAME="natural_language.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_MEDS_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"
    PRE_TRAINED_TOKENIZER="${HF_MODEL_NAME}"
    TOKENIZER_ARG="${HF_MODEL_NAME}"
fi

mkdir -p "${OUTPUT_DIR}"

uv run pre_training_tokenize \
    --config-name="${CONFIG_NAME}" \
    tokenizer.pre_trained_tokenizer="${PRE_TRAINED_TOKENIZER}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    num_workers="${NUM_WORKERS}"

DEBUG_OUTPUT_DIR="${DEBUG_DIR}/pretraining/${MODE}"
mkdir -p "${DEBUG_OUTPUT_DIR}"
echo "Outputting debug files to ${DEBUG_OUTPUT_DIR}"

uv run python mimic_struct_predict/debug/pretrain_output.py \
    --output_dir "${DEBUG_OUTPUT_DIR}" \
    --raw_data_dir "${OUTPUT_DIR}/raw_data/train/" \
    --tokenizer_dir "${TOKENIZER_ARG}" \
    --tokenized_dataset_dir "${OUTPUT_DIR}/tokenized_dataset"
