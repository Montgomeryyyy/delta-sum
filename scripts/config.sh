#!/usr/bin/env bash
# config.sh - Shared project configuration
# Have to define DATA_DIR and OUTPUT_DIR in terminal

DATASET_NAME="mimic-iv-demo"
DATASET_NAME_ED="mimic-iv-ed-demo"
MIMIC_VERSION="2.2"

PHYSIONET_URL="https://physionet.org/files"

MIMICIV_RAW_DIR="${DATA_DIR}/raw_input"
MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

NATURAL_LANGUAGE_INPUT_DIR="${DATA_DIR}/MEDS_cohort_with_headers"
STRUCTURED_INPUT_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers_and_quantiles"