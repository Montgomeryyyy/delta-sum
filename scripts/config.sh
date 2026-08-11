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
export DATA_DIR="${SCRATCH_DIR}/data"
export TRAINING_DIR="${SCRATCH_DIR}/training"
export INFERENCE_DIR="${SCRATCH_DIR}/inference"
export DEBUG_DIR="${SCRATCH_DIR}/debug"
export EVAL_PLOTS_DIR="${SCRATCH_DIR}/eval_plots"

export HF_MODEL_NAME="google/gemma-3-270m"

export MIMIC_VERSION="2.2"
export DATASET_NAME="mimiciv"
export MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"
export MIMICIV_PRE_MEDS_DIR="${DATA_DIR}/pre_MEDS"
export MIMICIV_MEDS_COHORT_DIR="${DATA_DIR}/MEDS_cohort"

export ACES_DEFINITIONS_DIR="./mimic_struct_predict/preprocessing/aces_task_definitions"
shopt -s nullglob
export ACES_TASKS=$(basename -a -s .yaml "${ACES_DEFINITIONS_DIR}"/*.yaml)
# export ACES_TASKS=hospital_length_of_stay_leq_7_days
shopt -u nullglob
export ACES_OUTPUT_DIR="${DATA_DIR}/aces_labels"

export NATURAL_LANGUAGE_MEDS_DIR="${MIMICIV_MEDS_COHORT_DIR}/add_delta_rows"
export NATURAL_LANGUAGE_DATA_DIR="${DATA_DIR}/natural_language"
export NATURAL_LANGUAGE_PRE_TRAINING_DIR="${NATURAL_LANGUAGE_DATA_DIR}/pre_training"
export NATURAL_LANGUAGE_TASK_DATA_DIR="${NATURAL_LANGUAGE_DATA_DIR}/tasks"
export NATURAL_LANGUAGE_TASK_EMBEDDING_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/task_embeddings"

export EHR_TEXT_DIR="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/ehr_texts"
export NATURAL_LANGUAGE_MODEL_DIR="${TRAINING_DIR}/natural_language"
export NATURAL_LANGUAGE_INFERENCE_DIR="${INFERENCE_DIR}/natural_language"
export NATURAL_LANGUAGE_LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/embedding_lr/models"
export NATURAL_LANGUAGE_LR_INFERENCE_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/embedding_lr"

export STRUCTURED_DATA_DIR="${DATA_DIR}/structured"
export STRUCTURED_MEDS_DIR="${MIMICIV_MEDS_COHORT_DIR}/data" # Output of bin_numeric_values
export STRUCTURED_PRE_TRAINING_DIR="${STRUCTURED_DATA_DIR}/pre_training"
export STRUCTURED_TASK_DATA_DIR="${STRUCTURED_DATA_DIR}/tasks"

export STRUCTURED_MODEL_DIR="${TRAINING_DIR}/structured"
export STRUCTURED_INFERENCE_DIR="${INFERENCE_DIR}/structured"
export STRUCTURED_LR_MODEL_DIR="${STRUCTURED_MODEL_DIR}/embedding_lr/models"
export STRUCTURED_LR_INFERENCE_DIR="${STRUCTURED_INFERENCE_DIR}/embedding_lr"


export MEDS_TAB_MEDS_DIR="${MIMICIV_MEDS_COHORT_DIR}/merge_to_MEDS_cohort"
export MEDS_TAB_OUTPUT_DATA_DIR="${DATA_DIR}/MEDS_tab"
export MEDS_TAB_OUTPUT_MODEL_DIR="${TRAINING_DIR}/MEDS_tab"

# ---- Enviorment Variables ----
CACHE_DIR="${SCRATCH_DIR}/../.cache"
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

export HF_HOME="${CACHE_DIR}/hf"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"

export TRACKIO_DIR="${TRAINING_DIR}/trackio"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_DIR}/.venv"


# Creates a tempdir so can mem offload hugginface operations
# But cache explodes in size (multiple copies of dataset) and don't care about fast
# reruns of maps (all are ~quick and should be run once). So delete cache on script exit
if [ -z "$HF_DATASETS_CACHE" ]; then
    mkdir -p "${CACHE_DIR}"
    export HF_DATASETS_CACHE=$(mktemp -d -p "${CACHE_DIR}" hf_datasets_cache_XXXXXX)
    # Ensure the cache is cleaned up when the script exits
    trap 'rm -rf "$HF_DATASETS_CACHE"' EXIT
fi
