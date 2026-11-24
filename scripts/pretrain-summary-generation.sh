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
#
#   # Shard patients across GPUs with a job array (configs that define
#   # num_shards/shard_id read them from SLURM_ARRAY_TASK_COUNT/ID). With
#   # --gres=gpu:2, tensor_parallel_size follows NUM_GPUS. Then merge on a CPU node:
#   sbatch --array=0-7 --gres=gpu:1 --constraint=h100 ... \
#     --output="$SCRATCH_DIR/../logs/summarize/%x-%A_%a.out" \
#     scripts/pretrain-summary-generation.sh normal summarize_medgemma_hosp_transfer admitted_only=true
#   scripts/pretrain-summary-generation.sh normal summarize_medgemma_hosp_transfer \
#     num_shards=8 merge_shards=true
#
#   # Remote ARC API configs (model.backend=arc) do not need a GPU:
#   sbatch \
#     --job-name=summarize-arc-lite \
#     --partition=slam_cpu \
#     --time=4:00:00 \
#     --ntasks=1 \
#     --cpus-per-task=8 \
#     --mem=32G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/summarize/%x-%j.out" \
#     scripts/pretrain-summary-generation.sh delta delta_only_summarize_arc_lite_hosp_transfer \
#     output_dir="$SCRATCH_DIR/data/natural_language/pre_training/arc_lite_delta_only_hosp_transfer"
#
#   Requires ARC_AI_API_KEY in the environment (e.g. from .env).

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
# Avoid "Cannot re-initialize CUDA in forked subprocess" when vLLM starts EngineCore.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

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
    # Named yaml owns model.model unless DOMAIN_ADAPT_MODEL_SUM is set; trailing
    # hydra overrides (model.model=...) still win either way.
    if [[ -n "${DOMAIN_ADAPT_MODEL_SUM}" ]]; then
        MODEL_ARGS=(model.model="${DOMAIN_ADAPT_MODEL_SUM}")
    else
        MODEL_ARGS=()
    fi
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
