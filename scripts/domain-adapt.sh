#!/usr/bin/env bash
# Domain-adapt a model to the raw EHR text with LoRA, then save a merged copy.
#
# Usage:
#   scripts/domain-adapt.sh [--model <hf_id_or_path>] [--input-dir <raw_text_dir>] \
#       [--output-dir <dir>] [--config-name <name>] [extra hydra overrides...]
#
#   --model        Model to adapt: a local directory, or an HF id already downloaded into HF_HOME
#                  (the hub is not contacted). Defaults to model.name in the config.
#   --input-dir    Raw text with <split>/*.parquet and an ehr_text column.
#                  Defaults to NATURAL_LANGUAGE_RAW_DATA_DIR (pretrain-meds-to-text.sh output).
#   --output-dir   Defaults to ${DOMAIN_ADAPT_MODEL_DIR}/<model name>.
#   --config-name  Config under mimic_struct_predict/models/config/domain_adapt/ (default adapt_medgemma).
#
# The merged model ends up in <output-dir>/<timestamp>/merged_model. Pass that to summarization:
#   scripts/pretrain-summary-generation.sh normal summarize_medgemma_hosp_transfer \
#       model.model=<output-dir>/<timestamp>/merged_model
#
# Example (27B in bf16 needs an 80GB GPU per process; add model.load_in_4bit=true for less):
#   sbatch \
#     --job-name=domain-adapt \
#     --partition=gpu_partition \
#     --time=2-00:00:00 \
#     --gres=gpu:1 \
#     --constraint=h100 \
#     --cpus-per-task=18 \
#     --mem=200G \
#     --chdir="$(pwd)" \
#     --export=ALL \
#     --output="$SCRATCH_DIR/../logs/domain_adapt/%x-%j.out" \
#     scripts/domain-adapt.sh --model google/medgemma-27b-it max_train_patients=20000
#
#   # Smoke test:
#   scripts/domain-adapt.sh max_train_patients=50 max_eval_patients=20 merge=false training_args.max_steps=5

set -euo pipefail

source "./scripts/config.sh"

MODEL=""
INPUT_DIR="${NATURAL_LANGUAGE_RAW_DATA_DIR}"
OUTPUT_DIR=""
CONFIG_NAME="adapt_medgemma"
HYDRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config-name) CONFIG_NAME="$2"; shift 2 ;;
        *) HYDRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! -f "./mimic_struct_predict/models/config/domain_adapt/${CONFIG_NAME}.yaml" ]]; then
    echo "Config not found: ${CONFIG_NAME}" >&2
    exit 1
fi

MODEL_ARGS=()
if [[ -n "${MODEL}" ]]; then
    MODEL_ARGS=(model.name="${MODEL}")
    MODEL_LABEL="$(basename "${MODEL}")"
else
    MODEL_LABEL="${CONFIG_NAME}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${DOMAIN_ADAPT_MODEL_DIR}/${MODEL_LABEL}}"

echo "Domain adaptation config=${CONFIG_NAME} model=${MODEL:-<from config>}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

export NCCL_P2P_DISABLE=1
# Compute nodes have no internet: load the model from a local path or the HF_HOME cache only.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

uv run accelerate launch \
    --config_file mimic_struct_predict/models/config/train/accelerate.yaml \
    --num_processes "${NUM_GPUS}" \
    -m mimic_struct_predict.models.domain_adapt \
    --config-name "${CONFIG_NAME}" \
    input_dir="${INPUT_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    "${MODEL_ARGS[@]}" \
    "${HYDRA_ARGS[@]}"
