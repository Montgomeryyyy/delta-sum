#!/usr/bin/env bash
# config.sh - Shared project configuration
# Have to define DATA_DIR and HF_MODEL_NAME in terminal

# ---- Check SCRATCH_DIR and HF_MODEL_NAME already defined ----
if [ -z "$SCRATCH_DIR" ]; then
    echo "Error: SCRATCH_DIR environment variable must be set." >&2
    exit 1
fi
if [ -z "$HF_MODEL_NAME" ]; then
    echo "Error: HF_MODEL_NAME environment variable must be set." >&2
    exit 1
fi

# ---- Internal Script Variables (No export needed) ----
DATASET_NAME="mimic-iv-demo"
DATASET_NAME_ED="mimic-iv-ed-demo"
MIMIC_VERSION="2.2"

PHYSIONET_URL="https://physionet.org/files"

DATA_DIR="${SCRATCH_DIR}/data"
TRAINING_DIR="${SCRATCH_DIR}/training"
INFERENCE_DIR="${SCRATCH_DIR}/inference"

PRETRAINED_MODEL_PATH="${SCRATCH_DIR}/hf_models/${HF_MODEL_NAME}"

MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
ACES_OUTPUT_DIR="${DATA_DIR}/aces_labels"

NATURAL_LANGUAGE_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers"
NATURAL_LANGUAGE_PROMPT_DATA_DIR="${DATA_DIR}/natural_language/prompts"
NATURAL_LANGUAGE_MODEL_DIR="${TRAINING_DIR}/natural_language"
NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"


STRUCTURED_INPUT_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers_and_quantiles"

# ---- Global Environment Exports (Read by external tools/Python) ----
export HF_HOME="${SCRATCH_DIR}/.cache/hf"
export TRACKIO_DIR="${SCRATCH_DIR}/trackio"
