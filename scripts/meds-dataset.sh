#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

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

echo "Outputting subjects to markdown for debugging at ${DEBUG_DIR}/meds"
uv run python mimic_struct_predict/debug/med_output.py \
    --output_dir "${DEBUG_DIR}/meds" \
    --meds_train_dir "${MIMICIV_MEDS_COHORT_DIR}/data/train/"

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

    echo "Outputting task ${task} to markdown for debugging at ${DEBUG_DIR}/aces/${task}"
    uv run python mimic_struct_predict/debug/aces_output.py \
        --output_dir "${DEBUG_DIR}/aces/${task}" \
        --aces_train_dir "${ACES_OUTPUT_DIR}/${task}/train/" \
        --meds_train_dir "${MEDS_TAB_MEDS_DIR}/train/"
done

echo "MEDS dataset pipeline completed successfully!"
