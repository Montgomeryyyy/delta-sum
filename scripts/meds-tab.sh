#!/usr/bin/env bash
# Based on https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/MIMICIV_TUTORIAL/task_tabularize_meds.sh


# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

N_WORKERS=$(nproc)

echo "  -> Generating Hydra subsets"
WINDOW_SIZES="[1d,7d,30d,365d,full]"
AGGS="[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
WINDOW_SUBSETS=$(uv run generate-subsets "${WINDOW_SIZES}")
AGG_SUBSETS=$(uv run generate-subsets "${AGGS}")

echo "Describing MEDS tabular data..."
uv run meds-tab-describe \
    "input_dir=${MEDS_TAB_MEDS_DIR}" \
    "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}"

echo "Running for tasks: ${ACES_TASKS}"
for task in ${ACES_TASKS}; do
    echo "Processing task: ${task}"
    mkdir -p "${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}"
    rsync -r "${MEDS_TAB_OUTPUT_DATA_DIR}/metadata/" "${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}/metadata"

    # We do not have static evennts (codes with no timestamp) so skip this step
    # echo "Tabularizing static data"
    # meds-tab-tabularize-static \
    #     "input_dir=${MIMICIV_MEDS_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR/${TASK}" \
    #     do_overwrite=False "input_label_dir=${TASKS_DIR}/${TASK}/" "$@"

    echo "  -> Running task_specific_tabularization"
    uv run meds-tab-tabularize-time-series \
        --multirun \
        worker="range(0,${N_WORKERS})" \
        hydra/launcher=joblib \
        "input_dir=${MEDS_TAB_MEDS_DIR}" \
        "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}" \
        "input_label_dir=${ACES_OUTPUT_DIR}/${task}/" \
        "do_overwrite=False" \
        "tabularization.min_code_inclusion_count=10" \
        "tabularization.window_sizes=${WINDOW_SIZES}" \
        "tabularization.aggs="${AGGS}""

    echo "  -> Running xgboost"
    # Explicitly setting default hyperparamters from https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/f506448c0dc2e9e29fc95120c6b2875397933451/src/MEDS_tabular_automl/configs/model_launcher/xgboost.yaml#L25
    uv run meds-tab-model \
        --multirun \
        model_launcher=xgboost \
        "input_dir=${MEDS_TAB_MEDS_DIR}" \
        "output_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}" \
        "input_tabularized_cache_dir=${MEDS_TAB_OUTPUT_DATA_DIR}/training/${task}/tabularize/" \
        "input_label_cache_dir=${ACES_OUTPUT_DIR}/${task}/" \
        "output_model_dir=${MEDS_TAB_OUTPUT_MODEL_DIR}/${task}/" \
        "task_name=${task}" \
        "do_overwrite=False" \
        "hydra.sweeper.n_trials=100" \
        "hydra.sweeper.n_jobs=8" \
        "model_launcher.model.nthread=10" \
        "+model_launcher.model.eta=tag(log, interval(0.001, 1))" \
        "+model_launcher.model.lambda=tag(log, interval(0.001, 1))" \
        "+model_launcher.model.alpha=tag(log, interval(0.001, 1))" \
        "+model_launcher.model.subsample=interval(0.5, 1)" \
        "+model_launcher.model.min_child_weight=interval(1e-2, 100)" \
        "+model_launcher.model.max_depth=range(2, 16)" \
        "model_launcher.training_params.num_boost_round=range(100, 1000)" \
        "model_launcher.training_params.early_stopping_rounds=range(1, 10)" \
        "tabularization.min_code_inclusion_count=tag(log, range(10, 1000000))" \
        "tabularization.window_sizes=${WINDOW_SUBSETS}" \
        "tabularization.aggs=${AGG_SUBSETS}" \
        "delete_below_top_k=1"
done
