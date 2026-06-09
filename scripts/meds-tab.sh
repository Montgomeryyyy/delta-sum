#!/usr/bin/env bash
# Based on https://meds-tab.readthedocs.io/en/latest/prediction/


# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config_full.sh"

# --- Pipeline Execution ---

# Print commands
set -x

N_WORKERS=$(nproc)

echo "  -> Generating Hydra subsets"
WINDOW_SIZES="[1d,7d,30d,365d, full]"
AGGS="[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
WINDOW_SUBSETS=$(uv run generate-subsets "${WINDOW_SIZES}")
AGG_SUBSETS=$(uv run generate-subsets "${AGGS}")

echo "Describing MEDS tabular data..."
uv run meds-tab-describe \
    "input_dir=${MIMICIV_MEDS_COHORT_DIR}/data" \
    "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}"

for task in ${ACES_TASKS}; do
    echo "Processing task: ${task}"
    mkdir -p "${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}"
    rsync -r "${MEDS_TAB_OUTPUT_DATA_DIR}/metadata/" "${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}/metadata"

    echo "  -> Running task_specific_tabularization"
    uv run meds-tab-tabularize-time-series \
        --multirun \
        worker="range(0,${N_WORKERS})" \
        hydra/launcher=joblib \
        "input_dir=${MIMICIV_MEDS_COHORT_DIR}/data" \
        "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}" \
        "input_label_dir=${ACES_OUTPUT_DIR}/${task}/" \
        "do_overwrite=False" \
        "tabularization.min_code_inclusion_count=10" \
        "tabularization.window_sizes=${WINDOW_SIZES}" \
        "tabularization.aggs="${AGGS}""

    echo "  -> Running xgboost"
    uv run meds-tab-model \
        --multirun \
        model_launcher=xgboost \
        "input_dir=${MIMICIV_MEDS_COHORT_DIR}/data" \
        "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}" \
        "input_tabularized_cache_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}/tabularize/" \
        "input_label_cache_dir=${ACES_OUTPUT_DIR}/${task}/" \
        "output_model_dir=${MEDS_TAB_OUTPUT_MODEL_DIR}/${task}/" \
        "task_name=${task}" \
        "do_overwrite=False" \
        "hydra.sweeper.n_trials=1000" \
        "hydra.sweeper.n_jobs=${N_WORKERS}" \
        "tabularization.min_code_inclusion_count=10" \
        "tabularization.window_sizes=${WINDOW_SUBSETS}" \
        "tabularization.aggs=${AGG_SUBSETS}" \
        "delete_below_top_k=1"
done
