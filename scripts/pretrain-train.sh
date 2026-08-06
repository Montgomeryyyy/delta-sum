#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

MODE=${1}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" && "$MODE" != "natural_language_lora" ]]; then
    echo "Usage: $0 [structured|natural_language|natural_language_lora]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

if [[ "$MODE" == "structured" ]]; then
    CONFIG_NAME="structured.yaml"
    DATASET_DIR="${STRUCTURED_PRE_TRAINING_DIR}/tokenized_dataset"
    TOKENIZER_DIR="${STRUCTURED_PRE_TRAINING_DIR}/tokenizer"
    OUTPUT_DIR="${STRUCTURED_MODEL_DIR}/pretraining"
elif [[ "$MODE" == "natural_language" ]]; then
    CONFIG_NAME="natural_language.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/tokenized_dataset"
    TOKENIZER_DIR="${HF_MODEL_NAME}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/pretraining"
elif [[ "$MODE" == "natural_language_lora" ]]; then
    CONFIG_NAME="natural_language_lora.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/tokenized_dataset"
    TOKENIZER_DIR="${HF_MODEL_NAME}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/pretraining_lora"
fi

# (P2P) over PCIe is broken or blocked by the BIOS/IOMMU
export NCCL_P2P_DISABLE=1

uv run accelerate launch \
    --config_file mimic_struct_predict/models/pre_training/config/train/accelerate.yaml \
    --num_processes "${NUM_GPUS}" \
    --no_python pre_training_train \
    --config-name "${CONFIG_NAME}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    model.name="${HF_MODEL_NAME}" \
    tokenizer_dir="${TOKENIZER_DIR}"
