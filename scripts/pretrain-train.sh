#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

MODE=${1}
shift || true

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" && "$MODE" != "natural_language_lora" ]]; then
    echo "Usage: $0 <mode> [options]"
    echo "Modes: structured | natural_language | natural_language_lora"
    echo "Options:"
    echo "  --resume-from-checkpoint <path>   Path to checkpoint directory to resume from"
    echo "  [...extra hydra overrides...]"
    exit 1
fi

HYDRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --resume-from-checkpoint)
            HYDRA_ARGS+=("resume_from_checkpoint=$2")
            shift 2
            ;;
        *)
            # Forward any unknown arguments (like Hydra overrides) directly
            HYDRA_ARGS+=("$1")
            shift 1
            ;;
    esac
done

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
    --config_file mimic_struct_predict/models/config/train/accelerate.yaml \
    --num_processes "${NUM_GPUS}" \
    --no_python pre_training_train \
    --config-name "${CONFIG_NAME}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    model.name="${HF_MODEL_NAME}" \
    tokenizer_dir="${TOKENIZER_DIR}" \
    "${HYDRA_ARGS[@]}"
