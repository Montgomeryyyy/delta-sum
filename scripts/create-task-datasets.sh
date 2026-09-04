#!/usr/bin/env bash

# Usage:
#   ./scripts/pretrain-dataset.sh [MODE]
#
# Arguments:
#   MODE: Either 'natural_language' (default) or 'structured'.
#         Determines which configuration and output directory are used.
#   TOKENIZER_DIR: Optional. Path to the model directory for tokenizer. Defaults to HF_MODEL_NAME.

# Exit immediately if a command exits with a non-zero status
set -e

MODE=${1}
TOKENIZER_DIR=${2}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language] [TOKENIZER_DIR]"
    exit 1
fi
if [ -z "$TOKENIZER_DIR" ]; then
    echo "Error: TOKENIZER_DIR argument not provided."
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

DEBUG_ARGS=""
if [[ "$MODE" == "structured" ]]; then
    CONFIG_NAME="structured.yaml"
    MEDS_DIR="${STRUCTURED_MEDS_DIR}"
    OUTPUT_DIR="${STRUCTURED_TASK_DATA_DIR}"
    DEBUG_OUTPUT_DIR="${DEBUG_DIR}/tasks/structured"
    DEBUG_ARGS="--structured"
else
    CONFIG_NAME="natural_language.yaml"
    MEDS_DIR="${NATURAL_LANGUAGE_MEDS_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
    DEBUG_OUTPUT_DIR="${DEBUG_DIR}/tasks/natural_language"
fi

mkdir -p "${OUTPUT_DIR}"

for TASK_NAME in $ACES_TASKS; do
    echo "Processing task: ${TASK_NAME}"
    uv run create_task_datasets \
        --config-name="${CONFIG_NAME}" \
        meds_dir="${MEDS_DIR}" \
        aces_dir="${ACES_OUTPUT_DIR}" \
        output_dir="${OUTPUT_DIR}" \
        tokenizer.pretrained_model_name_or_path="${TOKENIZER_DIR}" \
        num_workers="${NUM_WORKERS}" \
        "tasks=[${TASK_NAME}]"

    # Aggressively delete HF_DATASETS_CACHE and full text directory to prevent disk space exhaustion
    echo "Deleting ${HF_DATASETS_CACHE} and ${OUTPUT_DIR}/full_text/${TASK_NAME}"
    rm -rf "${HF_DATASETS_CACHE}"
    rm -rf "${OUTPUT_DIR}/full_text/${TASK_NAME}"
done

echo "Outputting subjects to markdown for debugging at ${DEBUG_OUTPUT_DIR}"
uv run python mimic_struct_predict/debug/task_output.py \
    --task_dir "${OUTPUT_DIR}/tokenized" \
    --debug_dir "${DEBUG_OUTPUT_DIR}" \
    --tokenizer_dir "${TOKENIZER_DIR}" \
    ${DEBUG_ARGS}
