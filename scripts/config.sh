#!/usr/bin/env bash
# config.sh - Shared project configuration
# Have to define DATA_DIR and HF_MODEL_NAME in terminal

# ---- Check DATA_DIR and HF_MODEL_NAME already defined ----
if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR environment variable must be set." >&2
    exit 1
fi
if [ -z "$HF_MODEL_NAME" ]; then
    echo "Error: HF_MODEL_NAME environment variable must be set." >&2
    exit 1
fi

# ---- Define config ----
DATASET_NAME="mimic-iv-demo"
DATASET_NAME_ED="mimic-iv-ed-demo"
MIMIC_VERSION="2.2"

PHYSIONET_URL="https://physionet.org/files"

INPUT_DIR="${DATA_DIR}/input"
TRAINING_DIR="${DATA_DIR}/training"
INFERENCE_DIR="${DATA_DIR}/inference"

PRETRAINED_MODEL_PATH="${INPUT_DIR}/hf_models/${HF_MODEL_NAME}"
export HF_HOME="${DATA_DIR}/.cache/hf"

MIMICIV_RAW_DIR="${INPUT_DIR}/raw_input"
MIMICIV_PRE_MEDS_DIR="${INPUT_DIR}/pre_MEDS"
MIMICIV_MEDS_COHORT_DIR="${INPUT_DIR}/MEDS_cohort"

ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
ACES_OUTPUT_DIR="${INPUT_DIR}/aces_labels"

NATURAL_LANGUAGE_INPUT_DIR="${INPUT_DIR}/MEDS_cohort_with_headers"
NATURAL_LANGUAGE_PROMPT_DATA_DIR="${INPUT_DIR}/natural_language/prompts"
NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"


STRUCTURED_INPUT_DATA_DIR="${INPUT_DIR}/MEDS_cohort_with_headers_and_quantiles"

TRACKIO_DIR="${TRAINING_DIR}/trackio"
