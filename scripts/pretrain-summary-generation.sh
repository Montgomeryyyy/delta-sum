#!/usr/bin/env bash

# Usage:
#   # SCRATCH_DIR and HF_TOKEN must already be set, same as the other scripts.
#   # Pass the mode as the first argument (normal|delta).
#   # Optional second argument is a Hydra config name under
#   # mimic_struct_predict/models/config/summarize/ (no .yaml).
#   # If omitted, normal uses summarize and delta uses delta_summerize.
#   # A named config keeps its own model.model. Trailing key=value args still
#   # override either way. input_dir and output_dir still come from config.sh,
#   # because the yaml files leave those required.
#
#   sbatch \
#     --job-name=summarize-delta \
#     --partition=gpu_partition \
#     --time=2-12:00:00 \
#     --gres=gpu:1 \
#     --constraint=h100 \
#     --ntasks=1 \
#     --cpus-per-task=18 \
#     --mem=64G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/summarize/%x-%j.out" \
#     scripts/pretrain-summary-generation.sh delta
#
#   scripts/pretrain-summary-generation.sh delta delta_summerize_medgemma

set -euo pipefail

# --- Get config ---
source "./scripts/config.sh"

# --- Get summarization mode and optional config name ---
SUM_MODE="${1:-normal}"
shift || true

CONFIG_NAME=""
if [[ "${1:-}" == "--config-name" ]]; then
    CONFIG_NAME="${2:-}"
    if [[ -z "${CONFIG_NAME}" ]]; then
        echo "Usage: $0 <normal|delta> [config_name] [extra hydra overrides]" >&2
        exit 1
    fi
    shift 2
elif [[ -n "${1:-}" && "${1}" != *=* && "${1}" != -* ]]; then
    CONFIG_NAME="$1"
    shift
fi

HYDRA_ARGS=("$@")

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# System nvcc is CUDA 11.5 and cannot compile FlashInfer's H100 (compute_90a) kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0

case "$SUM_MODE" in
    delta)
        DEFAULT_CONFIG="delta_summerize"
        OUTPUT_DIR="${NATURAL_LANGUAGE_DELTA_SUMMARIZED_DATA_DIR}"
        ;;
    normal)
        DEFAULT_CONFIG="summarize"
        OUTPUT_DIR="${NATURAL_LANGUAGE_SUMMARIZED_DATA_DIR}"
        ;;
    *)
        echo "Usage: $0 <normal|delta> [config_name] [extra hydra overrides]" >&2
        exit 1
        ;;
esac

if [[ -n "${CONFIG_NAME}" ]]; then
    CONFIG_FILE="./mimic_struct_predict/models/config/summarize/${CONFIG_NAME}.yaml"
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "Config not found: ${CONFIG_FILE}" >&2
        exit 1
    fi
    # Leave model.model to the named yaml. A later hydra override can still change it.
    MODEL_ARGS=()
else
    CONFIG_NAME="${DEFAULT_CONFIG}"
    MODEL_ARGS=(model.model="${HF_SUMMARIZATION_MODEL_NAME}")
fi

CONFIG_ARGS=(--config-name "${CONFIG_NAME}")

echo "Summarization mode: $SUM_MODE"
echo "Config: ${CONFIG_NAME}"
echo "Output directory: $OUTPUT_DIR"

"$UV_PROJECT_ENVIRONMENT/bin/python" \
    -m mimic_struct_predict.models.summarize \
    "${CONFIG_ARGS[@]}" \
    input_dir="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/raw_data" \
    output_dir="${OUTPUT_DIR}" \
    "${MODEL_ARGS[@]}" \
    hydra.job.chdir=false \
    "${HYDRA_ARGS[@]}"
