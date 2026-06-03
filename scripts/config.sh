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
DATA_DIR="${SCRATCH_DIR}/data"
TRAINING_DIR="${SCRATCH_DIR}/training"
INFERENCE_DIR="${SCRATCH_DIR}/inference"

PRETRAINED_MODEL_PATH="${SCRATCH_DIR}/hf_models/${HF_MODEL_NAME}"

MIMIC_VERSION="2.2"
DATASET_NAME="mimic-iv-demo"
MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic_demo"
# DATASET_NAME="mimiciv"
# MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
ACES_OUTPUT_DIR="${DATA_DIR}/aces_labels"

NATURAL_LANGUAGE_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers"
NATURAL_LANGUAGE_PROMPT_DATA_DIR="${DATA_DIR}/natural_language/prompts"
EHR_TEXT_DIR="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/ehr_texts"
NATURAL_LANGUAGE_MODEL_DIR="${TRAINING_DIR}/natural_language"
NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"

STRUCTURED_INPUT_DATA_DIR="${DATA_DIR}/MEDS_cohort_with_headers_and_quantiles"

# ---- Enviorment Variables ----
export HF_HOME="${SCRATCH_DIR}/.cache/hf"
export TRACKIO_DIR="${TRAINING_DIR}/trackio"
export UV_CACHE_DIR="${SCRATCH_DIR}/../.cache/uv"
export VLLM_CACHE_ROOT="${SCRATCH_DIR}/../.cache/vllm"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"
