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
    OUTPUT_DIR="${STRUCTURED_PRE_TRAINING_DIR}"
    DEBUG_OUTPUT="${DEBUG_DIR}/structured_pretrain_text_output.md"
    pre_trained_tokenizer=null
else
    CONFIG_NAME="natural_language.yaml"
    OUTPUT_DIR="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"
    DEBUG_OUTPUT="${DEBUG_DIR}/natural_language_pretrain_output.md"
    pre_trained_tokenizer="${HF_MODEL_NAME}"
fi

mkdir -p "${OUTPUT_DIR}"

uv run pre_training_tokenize \
    --config-name="${CONFIG_NAME}" \
    tokenizer.pre_trained_tokenizer="${pre_trained_tokenizer}" \
    dataset_dir="${MIMICIV_MEDS_COHORT_DIR}/data" \
    output_dir="${OUTPUT_DIR}" \
    num_workers="${NUM_WORKERS}"

echo "Outputting one subject to markdown for debugging at ${DEBUG_OUTPUT}"
uv run python -c "
import polars as pl; \
from pathlib import Path; \
df = pl.read_parquet('${OUTPUT_DIR}/raw_data/train/'); \
Path('$DEBUG_OUTPUT').write_text(df['ehr_text'][0]) \
"

if [[ "$MODE" == "structured" ]]; then
    DEBUG_OUTPUT="${DEBUG_DIR}/structured_pretrain_tokenized_output.txt"
    echo "Outputting one tokenized subject to markdown for debugging at ${DEBUG_OUTPUT}"

    uv run python -c "
import datasets
from pathlib import Path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('${OUTPUT_DIR}/tokenizer')
dataset = datasets.load_dataset('${OUTPUT_DIR}/tokenized_dataset')
Path('${DEBUG_OUTPUT}').write_text(str(tokenizer.convert_ids_to_tokens(dataset['train'][0]['input_ids'])))
"
fi
