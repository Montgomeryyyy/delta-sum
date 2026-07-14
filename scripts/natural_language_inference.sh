#!/bin/bash

# Exit immediately if any command fails
set -e

module load cuda/12.2.1-gcc-13.2.0

# Print GPU status at launch
echo "Running on ${HOSTNAME}"
nvidia-smi
nvcc --version

# --- Get config ---
source "./scripts/config_full.sh"

# --- Parse CLI args ---
SPLIT="held_out"
MODEL_PATH="${HF_MODEL_NAME}"
USAGE_STRING="Usage: $0 --dataset-name [name] --config-name [name] [--lora-adaptor path] [--model-path path] [--split name]"


while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      PROMPT_DATASET_NAME="$2"   # Assign the next argument to the variable
      shift 2     # Move past the flag and its value
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
    --lora-adaptor)
      LORA_ADAPTOR="$2"
      shift 2
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
    echo "Error: The --dataset-name argument is required." >&2
    echo $USAGE_STRING >&2
    exit 1
fi
if [[ -z "$CONFIG_NAME" ]]; then
    echo "Error: The --config-name argument is required." >&2
    echo $USAGE_STRING >&2
    exit 1
fi

# --- Run the natural language inference task ---
set -x

# --- Determine unified Run ID ---
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Determine Available GPUs ---
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    N=${#GPU_IDS[@]}
else
    N=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((N-1))))
fi

echo "Detected $N GPUs. Spawning $N Python processes with RUN_ID: $RUN_TIMESTAMP"

# Create the run directory so we can store logs there
RUN_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/${PROMPT_DATASET_NAME}/${RUN_TIMESTAMP}"
mkdir -p "$RUN_DIR"

if [[ -n "$LORA_ADAPTOR" ]]; then
    echo "Merging LoRA adaptor $LORA_ADAPTOR into $MODEL_PATH..."
    MERGED_MODEL_DIR="$RUN_DIR/merged_model"
    uv run natural_language_merge_peft_adaptor \
        base_model="${MODEL_PATH}" \
        lora_adaptor="${LORA_ADAPTOR}" \
        out_dir="${MERGED_MODEL_DIR}"

    # Update MODEL_PATH to point to the newly merged model for inference
    MODEL_PATH="$MERGED_MODEL_DIR"
fi

for i in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$i]}
    export RANK=$i
    export WORLD_SIZE=$N
    export CUDA_VISIBLE_DEVICES=$gpu_id

    if [ "$RANK" -eq 0 ]; then
        echo "Spawning Worker Rank $RANK/$WORLD_SIZE on GPU $gpu_id (Logging to SLURM .out file)"
        uv run natural_language_inference \
            --config-name "${CONFIG_NAME}" \
            model.model="${MODEL_PATH}" \
            dataset_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/${PROMPT_DATASET_NAME}" \
            output_dir="${NATURAL_LANGUAGE_INFERENCE_DIR}/${PROMPT_DATASET_NAME}" \
            hydra.run.dir="$RUN_DIR" \
            split="${SPLIT}" &
    else
        echo "Spawning Worker Rank $RANK/$WORLD_SIZE on GPU $gpu_id (Logging to $RUN_DIR/worker_${RANK}.log)"
        uv run natural_language_inference \
            --config-name "${CONFIG_NAME}" \
            model.model="${MODEL_PATH}" \
            dataset_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/${PROMPT_DATASET_NAME}" \
            output_dir="${NATURAL_LANGUAGE_INFERENCE_DIR}/${PROMPT_DATASET_NAME}" \
            hydra.run.dir="$RUN_DIR" \
            split="${SPLIT}" > "$RUN_DIR/worker_${RANK}.log" 2>&1 &
    fi
done

echo "Waiting for all $N workers to finish..."
wait
echo "All workers completed!"

if [[ -n "$LORA_ADAPTOR" ]]; then
    echo "Cleaning up merged model at $MERGED_MODEL_DIR..."
    rm -rf "$MERGED_MODEL_DIR"
fi
