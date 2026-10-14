#!/usr/bin/env bash

# Usage:
#   # SCRATCH_DIR and HF_TOKEN must already be set, same as the other scripts.
#   # Pass the mode as the first argument (normal|delta).
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

set -euo pipefail

# --- Get config ---
source "./scripts/config.sh"

# --- Get summarization mode ---
SUM_MODE="${1:-normal}"
shift || true

HYDRA_ARGS=("$@")

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
# System nvcc is CUDA 11.5 and cannot compile FlashInfer's H100 (compute_90a) kernels.
export VLLM_USE_FLASHINFER_SAMPLER=0

case "$SUM_MODE" in
    delta)
        CONFIG_ARGS=(--config-name delta_summerize)
        OUTPUT_DIR="${NATURAL_LANGUAGE_DELTA_SUMMARIZED_DATA_DIR}"
        ;;
    normal)
        CONFIG_ARGS=()
        OUTPUT_DIR="${NATURAL_LANGUAGE_SUMMARIZED_DATA_DIR}"
        ;;
    *)
        echo "Usage: $0 <normal|delta> [extra hydra overrides]"
        exit 1
        ;;
esac

# Same log location as the sbatch --output path above. Interactive runs were
# only printing to the terminal; this also keeps a copy on disk.
LOG_DIR="${SCRATCH_DIR}/../logs/summarize"
mkdir -p "${LOG_DIR}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    LOG_FILE="${LOG_DIR}/summarize-${SUM_MODE}-${SLURM_JOB_ID}.log"
else
    LOG_FILE="${LOG_DIR}/summarize-${SUM_MODE}-$(date +%Y%m%d_%H%M%S).log"
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Summarization mode: $SUM_MODE"
echo "Output directory: $OUTPUT_DIR"

"$UV_PROJECT_ENVIRONMENT/bin/python" \
    -m mimic_struct_predict.models.summarize \
    "${CONFIG_ARGS[@]}" \
    input_dir="${NATURAL_LANGUAGE_PRE_TRAINING_DIR}/raw_data" \
    output_dir="${OUTPUT_DIR}" \
    model.model="${HF_SUMMARIZATION_MODEL_NAME}" \
    hydra.job.chdir=false \
    "${HYDRA_ARGS[@]}"
