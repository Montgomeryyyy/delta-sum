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
    CONFIG_NAME="create_task_datasets_structured.yaml"
    OUTPUT_DIR="${STRUCTURED_TASK_DATA_DIR}"
    DEBUG_OUTPUT="${DEBUG_DIR}/structured_pretrain_task_output.md"
else
    CONFIG_NAME="create_task_datasets_nl.yaml"
    OUTPUT_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}"
    DEBUG_OUTPUT="${DEBUG_DIR}/natural_language_task_output.md"
fi

mkdir -p "${OUTPUT_DIR}"

uv run create_task_datasets \
    --config-name="${CONFIG_NAME}" \
    meds_dir="${MIMICIV_MEDS_COHORT_DIR}/data" \
    aces_dir="${ACES_OUTPUT_DIR}" \
    output_dir="${OUTPUT_DIR}"

echo "${OUTPUT_DIR}"

echo "Outputting one subject to markdown for debugging at ${DEBUG_OUTPUT}"
uv run python -c "
import polars as pl; \
from pathlib import Path; \
df = pl.read_parquet(next(Path('${OUTPUT_DIR}').glob('*/train/*.parquet'))); \
Path('$DEBUG_OUTPUT').write_text(df['ehr_text'][0]) \
"
