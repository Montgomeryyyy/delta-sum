#!/usr/bin/env bash

# Usage:
#   ./scripts/pre-train-generation.sh [MODE] [MODEL_DIR] [OUTPUT_DIRECTORY_NAME] [--split SPLIT]
#
# Arguments:
#   MODE: Either 'natural_language' or 'structured'.
#   MODEL_DIR: The directory or huggingface hub name of the model to use.
#   OUTPUT_DIRECTORY_NAME: The name of the output directory (directory name in NATURAL_LANGUAGE_INFERENCE_DIR).
#   --split: Optional flag to specify the split (e.g. 'tuning', 'held_out'). Defaults to 'held_out'.

# Exit immediately if a command exits with a non-zero status
set -e

MODE=${1}
MODEL_DIR=${2}
OUTPUT_DIRECTORY_NAME=${3}

if [[ "$MODE" != "structured" && "$MODE" != "natural_language" ]]; then
    echo "Usage: $0 [structured|natural_language] [MODEL_DIR] [OUTPUT_DIRECTORY_NAME]"
    exit 1
fi
if [ -z "$MODEL_DIR" ]; then
    echo "Error: MODEL_DIR argument not provided."
    exit 1
fi
if [ -z "$OUTPUT_DIRECTORY_NAME" ]; then
    echo "Error: OUTPUT_DIRECTORY_NAME argument not provided."
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

module load cuda/12.2.1-gcc-13.2.0

# --- Pipeline Execution ---

# --- Determine Available GPUs ---
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS="," read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    N=${#GPU_IDS[@]}
else
    N=$(nvidia-smi -L | wc -l)
    GPU_IDS=($(seq 0 $((N-1))))
fi

SPLIT="tuning"
for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == "--split" ]]; then
        next=$((i+1))
        SPLIT="${!next}"
    fi
done

if [[ "$MODE" == "structured" ]]; then
    TASK_DATASET_DIR="${STRUCTURED_TASK_DATA_DIR}/tokenized"
    OUTPUT_DIR="${STRUCTURED_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/rollout"
    CONFIG_NAME="structured.yaml"
else
    TASK_DATASET_DIR="${NATURAL_LANGUAGE_TASK_DATA_DIR}/tokenized"
    OUTPUT_DIR="${NATURAL_LANGUAGE_INFERENCE_DIR}/${OUTPUT_DIRECTORY_NAME}/rollout"
    CONFIG_NAME="natural_language.yaml"
fi

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

WORKERS_PER_GPU=$(( NUM_WORKERS / N ))
if [ "$WORKERS_PER_GPU" -lt 1 ]; then WORKERS_PER_GPU=1; fi

echo "--- Running vLLM Inference ---"
echo "Found tasks: ${ACES_TASKS}"

for task in ${ACES_TASKS}; do
    echo "Starting task ${task}..."
    task_dir="${TASK_DATASET_DIR}/${task}"

    for i in "${!GPU_IDS[@]}"; do
        gpu_id=${GPU_IDS[$i]}
        export RANK=$i
        export WORLD_SIZE=$N
        export CUDA_VISIBLE_DEVICES=$gpu_id

        RUN_DIR="${OUTPUT_DIR}/${SPLIT}/${task}/${RUN_TIMESTAMP}"
        mkdir -p "$RUN_DIR"

        echo "Spawning Worker Rank $RANK/$WORLD_SIZE on GPU $gpu_id (Logging to $RUN_DIR/worker_${RANK}.log)"

        uv run pre_training_generative_inference \
            --config-name="${CONFIG_NAME}" \
            model.pretrained_model_name_or_path="${MODEL_DIR}" \
            dataset_dir="${task_dir}/${SPLIT}" \
            output_dir="${OUTPUT_DIR}/${task}" \
            task_prevalence_path="${TASK_PREVALENCES_PATH}" \
            num_workers="${WORKERS_PER_GPU}" \
            hydra.run.dir="$RUN_DIR" \
            task="${task}" > "$RUN_DIR/worker_${RANK}.log" 2>&1 &
    done

    echo "Waiting for all $N workers to finish..."
    wait
    echo "All workers completed!"

done
