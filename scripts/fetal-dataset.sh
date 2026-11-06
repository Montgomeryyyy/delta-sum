#!/usr/bin/env bash
# Run ACES task definitions on an existing MEDS dataset.
#
# Usage:
#   scripts/fetal-dataset.sh <meds_dir> [task ...]
#
#   <meds_dir>  MEDS dataset root (contains data/<split>/*.parquet and metadata/)
#   [task ...]  ACES definition names in ${ACES_DEFINITIONS_DIR} (without .yaml).
#               Defaults to every definition in that folder.
#
# Example:
#   scripts/fetal-dataset.sh /path/to/MEDS_cohort ptb first_time_ptb

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <meds_dir> [task ...]" >&2
    exit 1
fi

MEDS_DIR="$1"
shift
TASKS=["ptb", "first_time_ptb"] #"${*:-${ACES_TASKS}}"

if [ ! -d "${MEDS_DIR}/data" ]; then
    echo "Error: ${MEDS_DIR}/data not found." >&2
    exit 1
fi

# --- Pipeline Execution ---

# Print commands
# set -x

echo "Running ACES pipeline on ${MEDS_DIR} for tasks: ${TASKS}"
for task in ${TASKS}; do
    if [ ! -f "${ACES_DEFINITIONS_DIR}/${task}.yaml" ]; then
        echo "Error: ${ACES_DEFINITIONS_DIR}/${task}.yaml not found." >&2
        exit 1
    fi

    rm -rf "${ACES_OUTPUT_DIR:?}/${task}"
    uv run aces-cli \
        config_path="${ACES_DEFINITIONS_DIR}/${task}.yaml" \
        data.standard=meds \
        data=sharded \
        data.root="${MEDS_DIR}/data" \
        "data.shard=$(uv run expand_shards "${MEDS_DIR}/data")" \
        cohort_dir="${ACES_OUTPUT_DIR}" \
        cohort_name="${task}" \
        hydra/launcher=joblib \
        hydra.launcher.n_jobs="${NUM_WORKERS}" \
        --multirun

    echo "Outputting task ${task} to markdown for debugging at ${DEBUG_DIR}/aces/${task}"
    uv run python mimic_struct_predict/debug/aces_output.py \
        --output_dir "${DEBUG_DIR}/aces/${task}" \
        --aces_train_dir "${ACES_OUTPUT_DIR}/${task}/train/" \
        --meds_train_dir "${MEDS_DIR}/data/train/"
done

echo "ACES labelling completed successfully!"
