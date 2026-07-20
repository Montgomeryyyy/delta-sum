# config.sh - Shared project configuration
# NOTE: This script must be SOURCED, not executed: source config.sh

# ---- Check SCRATCH_DIR already defined ----
if [ -z "$SCRATCH_DIR" ]; then
    echo "Error: SCRATCH_DIR environment variable must be set." >&2
    return 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable must be set." >&2
    return 1
fi


# ---- Internal Script Variables (No export needed) ----
DATA_DIR="${SCRATCH_DIR}/data"
TRAINING_DIR="${SCRATCH_DIR}/training"
INFERENCE_DIR="${SCRATCH_DIR}/inference"
DEBUG_DIR="${SCRATCH_DIR}/debug"

HF_MODEL_NAME="google/gemma-3-270m-it"

MIMIC_VERSION="2.2"
DATASET_NAME="mimic-iv-demo"
MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
# DATASET_NAME="mimiciv"
# MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
export MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
ACES_OUTPUT_DIR="${DATA_DIR}/aces_labels"

NATURAL_LANGUAGE_PRE_TRAINING_DIR="${DATA_DIR}/natural_language/pre_training"
NATURAL_LANGUAGE_PROMPT_DATA_DIR="${DATA_DIR}/natural_language/prompts"
EHR_TEXT_DIR="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/ehr_texts"
NATURAL_LANGUAGE_MODEL_DIR="${TRAINING_DIR}/natural_language"
NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"

MEDS_TAB_OUTPUT_DATA_DIR="${DATA_DIR}/MEDS_tab"
MEDS_TAB_OUTPUT_MODEL_DIR="${TRAINING_DIR}/MEDS_tab"

# ---- Enviorment Variables ----
CACHE_DIR="${SCRATCH_DIR}/../.cache"
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

export HF_HOME="${CACHE_DIR}/hf"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"

export TRACKIO_DIR="${TRAINING_DIR}/trackio"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"
