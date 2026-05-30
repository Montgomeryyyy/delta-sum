# config.sh - Shared project configuration
# NOTE: This script must be SOURCED, not executed: source config.sh

# ---- Check SCRATCH_DIR and HF_MODEL_NAME already defined ----
if [ -z "$SCRATCH_DIR" ]; then
    echo "Error: SCRATCH_DIR environment variable must be set." >&2
    return 1
fi
if [ -z "$HF_MODEL_NAME" ]; then
    echo "Error: HF_MODEL_NAME environment variable must be set." >&2
    return 1
fi

# ---- Internal Script Variables (No export needed) ----
export DATASET_NAME="mimic-iv-demo"
export DATASET_NAME_ED="mimic-iv-ed-demo"
export MIMIC_VERSION="2.2"

export PHYSIONET_URL="https://physionet.org/files"

export DATA_DIR="${SCRATCH_DIR}/data"
export TRAINING_DIR="${SCRATCH_DIR}/training"
export INFERENCE_DIR="${SCRATCH_DIR}/inference"

export HF_HOME="${SCRATCH_DIR}/.cache/hf"
export PRETRAINED_MODEL_PATH="${SCRATCH_DIR}/hf_models/${HF_MODEL_NAME}"

export MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
export MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
export MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

export ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
export ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
export ACES_OUTPUT_DIR="${DATA_DIR}/aces_labels"

export NATURAL_LANGUAGE_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers"
export NATURAL_LANGUAGE_PROMPT_DATA_DIR="${DATA_DIR}/natural_language/prompts"
export NATURAL_LANGUAGE_MODEL_DIR="${TRAINING_DIR}/natural_language"
export NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"

export STRUCTURED_INPUT_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers_and_quantiles"

export TRACKIO_DIR="${TRAINING_DIR}/trackio"
export UV_CACHE_DIR="${SCRATCH_DIR}/../.cache/uv"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"
