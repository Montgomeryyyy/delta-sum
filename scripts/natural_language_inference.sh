#!/bin/bash

# Exit immediately if any command fails
set -e

module load cuda/12.2.1-gcc-13.2.0

# Print GPU status at launch
echo "Running on ${HOSTNAME}"
nvidia-smi
nvcc --version

# --- Get config ---
source "./scripts/config.sh"

# --- Parse CLI args ---
SPLIT="held_out"
MODEL_PATH="${PRETRAINED_MODEL_PATH}"
USAGE_STRING="Usage: $0 -d [dataset-name] -c [config-name]"


while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      PROMPT_DATASET_NAME="$2"   # Assign the next argument to the variable
      shift 2     # Move past the flag (-u) and its value (Alice)
      ;;
    --config-name)
      CONFIG_NAME="$2"   # Assign the next argument to the variable
      shift 2     # Move past the flag and its value
      ;;
    --model-path)
      MODEL_PATH="$2"   # Assign the next argument to the variable
      shift 2     # Move past the flag and its value
      ;;
    --split)
      SPLIT="$2"   # Assign the next argument to the variable
      shift 2     # Move past the flag and its value
      ;;
    -h|--help)
      echo $USAGE_STRING
      exit 0
      ;;
    *) # Handle unknown flags or positional arguments
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$PROMPT_DATASET_NAME" ]]; then
    echo "Error: The -u/--user argument is required." >&2
    echo $USAGE_STRING >&2
    exit 1
fi
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: The -c/--config-name argument is required." >&2
    echo $USAGE_STRING >&2
    exit 1
fi

# --- Run the natural language inference task ---
set -x

# rm -rf ${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/thinking_yes_no

# uv run natural_language_create_model_prompts \
#     model.pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
#     tokenization.output_format=yes_no \
#     tokenization.enable_thinking=true \
#     data_dir="${EHR_TEXT_DIR}" \
#     output_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/thinking_yes_no"

uv run natural_language_inference \
    --config-name "${CONFIG_NAME}" \
    model.pretrained_model_name_or_path="${MODEL_PATH}" \
    dataset_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/${PROMPT_DATASET_NAME}" \
    output_dir="${NATURAL_LANGUAGE_INFERENCE_DIR}/${PROMPT_DATASET_NAME}" \
    split="${SPLIT}"
