#!/usr/bin/env bash

# Usage:
#   ./scripts/pretrain-meds-to-text.sh [MODE]
#
# Arguments:
#   MODE: Either 'natural_language' (default) or 'structured'.
#         Determines which configuration and output directory are used.
#
# Converts MEDS shards to patient-level EHR text under
# ${OUTPUT_DIR}/raw_data. Run this before summarization or tokenization.

set -e

MODE=${1}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

if [[ "$MODE" == "structured" ]]; then
    CONFIG_NAME="structured.yaml"
    DATASET_DIR="${STRUCTURED_MEDS_DIR}"
    OUTPUT_DIR="${STRUCTURED_PRE_TRAINING_DIR}"
else
    CONFIG_NAME="natural_language.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_MEDS_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

uv run pre_training_meds_to_text \
    --config-name="${CONFIG_NAME}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}"
