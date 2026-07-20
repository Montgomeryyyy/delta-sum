#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
set -x

mkdir -p "${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"

uv run pre_training_tokenize \
    --config-name="natural_language.yaml" \
    tokenizer.pre_trained_tokenizer="${HF_MODEL_NAME}" \
    dataset_dir="${MIMICIV_MEDS_COHORT_DIR}/data" \
    output_dir="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}"

DEBUG_OUTPUT="${DEBUG_DIR}/natural_language_pretrain_output.md"

echo "Outputting one subject to markdown for debugging at ${DEBUG_OUTPUT}"
uv run python -c "
import polars as pl; \
from pathlib import Path; \
df = pl.read_parquet('${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/raw_data/train/'); \
Path('$DEBUG_OUTPUT').write_text(df['ehr_text'][0]) \
"
