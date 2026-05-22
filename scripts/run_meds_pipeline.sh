#!/usr/bin/env bash

#SBATCH --output=/scratch/users/%u/logs/run_meds_pipeline/%j.out

# To run need to set DATA_DIR
# sbatch -p cpu -t 5 -D $(pwd) --export=ALL --export=DATA_DIR scripts/run_meds_pipeline.sh

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

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
uv run MEDS_transform-pipeline \
    pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/ETL.yaml

echo "Running add headers pipeline..."
rm -rf "$NATURAL_LANGUAGE_INPUT_DIR"
INPUT_DIR="$MIMICIV_MEDS_COHORT_DIR" \
OUTPUT_DIR="$NATURAL_LANGUAGE_INPUT_DIR" \
DATASET_NAME="$DATASET_NAME" \
DATASET_VERSION="$MIMIC_VERSION" \
uv run MEDS_transform-pipeline \
    pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_headers.yaml

echo "Running add quantiles pipeline..."
rm -rf "$STRUCTURED_INPUT_DATA_DIR"
INPUT_DIR="$NATURAL_LANGUAGE_INPUT_DIR" \
OUTPUT_DIR="$STRUCTURED_INPUT_DATA_DIR" \
DATASET_NAME="$DATASET_NAME" \
DATASET_VERSION="$MIMIC_VERSION" \
uv run MEDS_transform-pipeline \
    pipeline_config_fp=./mimic_struct_predict/preprocessing/meds/config/add_quantiles.yaml

echo "MEDS dataset pipeline completed successfully!"