#!/bin/bash

# Exit immediately if a command exits with a non-zero status,
# and catch pipeline failures.
set -eo pipefail

# --- Get config ---
source "./scripts/config.sh"

module load cuda/12.2.1-gcc-13.2.0

# Print GPU status at launch
nvidia-smi
nvcc --version

# --- Parse CLI args ---
USAGE_STRING="Usage: $0 --dataset-name [dataset-name] --config-name [config-name]"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --dataset-name requires a non-empty argument." >&2
        exit 1
      fi
      PROMPT_DATASET_NAME="$2"
      shift 2
      ;;
    --config-name)
      if [[ -z "$2" || "$2" == -* ]]; then
        echo "Error: --config-name requires a non-empty argument." >&2
        exit 1
      fi
      CONFIG_NAME="$2"
      shift 2
      ;;
    -h|--help)
      echo "$USAGE_STRING"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "$USAGE_STRING" >&2
      exit 1
      ;;
  esac
done

# --- Validate Required Inputs ---
if [[ -z "$PROMPT_DATASET_NAME" ]]; then
    echo "Error: The --dataset-name argument is required." >&2
    echo "$USAGE_STRING" >&2
    exit 1
fi

if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: The --config-name argument is required." >&2
    echo "$USAGE_STRING" >&2
    exit 1
fi

# --- Run the natural language inference task ---
# Echo commands to stderr for debugging
set -x

uv run natural_language_grpo \
    --config-name "${CONFIG_NAME}" \
    model.pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    dataset_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/${PROMPT_DATASET_NAME}" \
    output_dir="${NATURAL_LANGUAGE_MODEL_DIR}/grpo/${PROMPT_DATASET_NAME}"
