#!/usr/bin/env bash

# Usage:
#   ./scripts/linear-probe-inference.sh [MODE] [MODEL_DIR]
#
# Arguments:
#   MODE: Either 'natural_language' or 'structured'.
#   MODEL_DIR: Optional. Path to the model directory. Defaults to HF_MODEL_NAME.
#   --zero-shot: Optional flag.
#   --split: Optional flag to specify the split (e.g. 'tuning', 'test'). Defaults to 'tuning'.

# Exit immediately if a command exits with a non-zero status
set -e

MODE=${1}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language] [MODEL_DIR]"
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

MODEL_DIR=${2:-${HF_MODEL_NAME}}

# --- Pipeline Execution ---

IS_ZERO_SHOT=false
SPLIT="held_out"
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--zero-shot" ]]; then
        IS_ZERO_SHOT=true
    elif [[ "${!i}" == "--split" ]]; then
        next=$((i+1))
        SPLIT="${!next}"
    fi
done

if [[ "$MODE" == "structured" ]]; then
    EMBEDDING_OUTPUT_DIR="${STRUCTURED_INFERENCE_DIR}/pt/task_embeddings"
    LR_MODEL_DIR="${STRUCTURED_LR_MODEL_DIR}"
    INFERENCE_OUTPUT_DIR="${STRUCTURED_LR_INFERENCE_DIR}"
else
    if [[ "$IS_ZERO_SHOT" == true ]]; then
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/zero_shot/task_embeddings"
        LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/zero_shot/embedding_lr/models"
        INFERENCE_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/zero_shot/embedding_lr_predictions"
    else
        EMBEDDING_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/cpt/task_embeddings"
        LR_MODEL_DIR="${NATURAL_LANGUAGE_MODEL_DIR}/cpt/embedding_lr/models"
        INFERENCE_OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/cpt/embedding_lr_predictions"
    fi
fi

echo "--- Running Logistic Regression Inference ---"
echo "Found tasks: $ACES_TASKS"
for task in $ACES_TASKS; do
    task_dir="${EMBEDDING_OUTPUT_DIR}/${task}"

    if [ ! -d "$task_dir" ]; then
        echo "Warning: Task directory not found at $task_dir. Skipping."
        continue
    fi
    echo "Running inference for task: $task split: $SPLIT"

    DATASET_PATH="${task_dir}/${SPLIT}/embeddings.parquet"
    echo "Found embeddings: $DATASET_PATH"

    # Locate the most recent multirun best_trial model.joblib
    LR_PATH=$(ls -1td "${LR_MODEL_DIR}/${task}"/multirun_*/best_trial/model.joblib 2>/dev/null | head -n 1 || true)

    if [ -z "$LR_PATH" ]; then
        echo "Error: No model found for $task at ${LR_MODEL_DIR}/${task}/multirun_*/best_trial/model.joblib"
        exit 1
    fi
    echo "Found LR model: ${LR_PATH}"

    uv run embedding_and_lr_inference_lr \
        dataset_path="${DATASET_PATH}" \
        output_dir="${INFERENCE_OUTPUT_DIR}/${SPLIT}/${task}" \
        lr_model_path="${LR_PATH}"
done
