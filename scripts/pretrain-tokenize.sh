#!/usr/bin/env bash

# Usage:
#   ./scripts/pretrain-tokenize.sh [MODE] [CHUNKED] [extra hydra overrides...]
#
# Arguments:
#   MODE:    'natural_language' (default) or 'structured'.
#   CHUNKED: optional. Pass 'chunked' to tokenize summarized windows in place
#            into tokenized_chunks_dataset (patient→chunks nesting preserved).
#            Default is classic patient-level tokenization into tokenized_dataset.
#   Extra:   optional Hydra key=value overrides (last wins), e.g.
#              input_dir=/path/to/summarized_dataset \
#              output_dir=/path/to/parent_dir
#
# Classic mode expects raw EHR text from pretrain-meds-to-text.sh
# (${OUTPUT_DIR}/raw_data). Chunked mode expects a summarized_dataset
# (override with SUMMARIZED_DATASET_DIR or input_dir=...).
#
# Example:
#   ./scripts/pretrain-tokenize.sh natural_language chunked \
#     input_dir="$SCRATCH_DIR/data/natural_language/pre_training/medgemma_delta_summarized_data_hosp_transfer/summarized_dataset" \
#     output_dir="$SCRATCH_DIR/data/natural_language/pre_training/medgemma_delta_tokenized_chunks"
#   (chunked: output_dir is the HF dataset path itself)

set -e

MODE=${1:-}
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 [structured|natural_language] [chunked] [extra hydra overrides...]"
    exit 1
fi
shift

CHUNKED_ARG=""
if [[ "${1:-}" == "chunked" ]]; then
    CHUNKED_ARG="chunked"
    shift
fi

HYDRA_ARGS=("$@")

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language] [chunked] [extra hydra overrides...]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

if [[ "$MODE" == "structured" ]]; then
    CONFIG_NAME="structured.yaml"
    OUTPUT_DIR="${STRUCTURED_PRE_TRAINING_DIR}"
    PRE_TRAINED_TOKENIZER=null
    TOKENIZER_ARG="${OUTPUT_DIR}/tokenizer"
    DEFAULT_SUMMARIZED_DIR="${STRUCTURED_PRE_TRAINING_DIR}/summarized_data/summarized_dataset"
else
    CONFIG_NAME="natural_language.yaml"
    OUTPUT_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"
    PRE_TRAINED_TOKENIZER="${HF_MODEL_NAME}"
    TOKENIZER_ARG="${HF_MODEL_NAME}"
    DEFAULT_SUMMARIZED_DIR="${NATURAL_LANGUAGE_SUMMARIZED_DATA_DIR}/summarized_dataset"
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "$CHUNKED_ARG" == "chunked" ]]; then
    INPUT_DIR="${SUMMARIZED_DATASET_DIR:-$DEFAULT_SUMMARIZED_DIR}"
    # Default dataset path; override with output_dir=... (no nested tokenized_chunks_dataset).
    CHUNKED_OUTPUT_DIR="${OUTPUT_DIR}/tokenized_chunks_dataset"
    echo "Chunked tokenization from ${INPUT_DIR} → ${CHUNKED_OUTPUT_DIR} (or output_dir override)"
    if [[ ${#HYDRA_ARGS[@]} -gt 0 ]]; then
        echo "Hydra overrides: ${HYDRA_ARGS[*]}"
    fi
    uv run pre_training_tokenize \
        --config-name="${CONFIG_NAME}" \
        tokenizer.pre_trained_tokenizer="${PRE_TRAINED_TOKENIZER}" \
        chunked=true \
        input_dir="${INPUT_DIR}" \
        output_dir="${CHUNKED_OUTPUT_DIR}" \
        num_workers="${NUM_WORKERS}" \
        "${HYDRA_ARGS[@]}"
else
    RAW_DATA_DIR="${OUTPUT_DIR}/raw_data"
    if [[ ! -d "${RAW_DATA_DIR}" ]] || ! compgen -G "${RAW_DATA_DIR}/*/*.parquet" > /dev/null; then
        echo "Raw data not found at ${RAW_DATA_DIR}; running MEDS -> text first."
        ./scripts/pretrain-meds-to-text.sh "${MODE}"
    fi

    echo "Classic tokenization from ${RAW_DATA_DIR} → ${OUTPUT_DIR}/tokenized_dataset"
    if [[ ${#HYDRA_ARGS[@]} -gt 0 ]]; then
        echo "Hydra overrides: ${HYDRA_ARGS[*]}"
    fi
    uv run pre_training_tokenize \
        --config-name="${CONFIG_NAME}" \
        tokenizer.pre_trained_tokenizer="${PRE_TRAINED_TOKENIZER}" \
        chunked=false \
        input_dir="${RAW_DATA_DIR}" \
        output_dir="${OUTPUT_DIR}" \
        num_workers="${NUM_WORKERS}" \
        "${HYDRA_ARGS[@]}"

    DEBUG_OUTPUT_DIR="${DEBUG_DIR}/pretraining/${MODE}"
    mkdir -p "${DEBUG_OUTPUT_DIR}"
    echo "Outputting debug files to ${DEBUG_OUTPUT_DIR}"

    uv run python mimic_struct_predict/debug/pretrain_output.py \
        --output_dir "${DEBUG_OUTPUT_DIR}" \
        --raw_data_dir "${OUTPUT_DIR}/raw_data/train/" \
        --tokenizer_dir "${TOKENIZER_ARG}" \
        --tokenized_dataset_dir "${OUTPUT_DIR}/tokenized_dataset"
fi
