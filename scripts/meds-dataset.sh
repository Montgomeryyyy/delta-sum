#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
set -x

echo "Creating pre-MEDS dataset from MIMIC-IV..."
rm -rf "$MIMICIV_PRE_MEDS_DIR"
uv run pre_MEDS \
    input_dir="$MIMICIV_RAW_DIR" \
    output_dir="$MIMICIV_PRE_MEDS_DIR" \
    do_overwrite=true \
    do_copy=true

echo "Running MEDS transform pipeline..."
rm -rf "$MIMICIV_MEDS_COHORT_DIR"

INPUT_DIR="$MIMICIV_PRE_MEDS_DIR" \
OUTPUT_DIR="$MIMICIV_MEDS_COHORT_DIR" \
DATASET_NAME="$DATASET_NAME" \
DATASET_VERSION="$MIMIC_VERSION" \
N_WORKERS="${NUM_WORKERS}" \
uv run MEDS_transform-pipeline \
    pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/ETL.yaml \

DEBUG_OUTPUT="${DEBUG_DIR}/meds_output.md"
mkdir -p "${DEBUG_DIR}"
echo "Outputting one subject to markdown for debugging at ${DEBUG_OUTPUT}"
uv run python -c "
import polars as pl; \
from pathlib import Path; \
df = pl.read_parquet('$MIMICIV_MEDS_COHORT_DIR/data/train/'); \
markdown_str = df.filter(pl.col('subject_id') == df['subject_id'][0]).to_pandas().to_markdown(index=False); \
Path('$DEBUG_OUTPUT').write_text(markdown_str) \
"

echo "Running ACES pipeline"
rm -rf "${ACES_OUTPUT_DIR}"
for task in ${ACES_TASKS}; do
    uv run aces-cli \
        config_path="${ACES_DEFINITIONS_DIR}/${task}.yaml" \
        data.standard=meds \
        data=sharded \
        data.root="${MIMICIV_MEDS_COHORT_DIR}/data" \
        "data.shard=$(uv run expand_shards "${MIMICIV_MEDS_COHORT_DIR}/data")" \
        cohort_dir="${ACES_OUTPUT_DIR}" \
        cohort_name="${task}" \
        hydra/launcher=joblib \
        hydra.launcher.n_jobs="${NUM_WORKERS}" \
        --multirun
done

echo "MEDS dataset pipeline completed successfully!"
