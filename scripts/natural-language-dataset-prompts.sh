#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Execute Commands ---

set -x

uv run natural_language_create_task_ehr_texts \
    meds_dir="${NATURAL_LANGUAGE_DATA_DIR}/data" \
    aces_dir="${ACES_OUTPUT_DIR}" \
    output_dir="${EHR_TEXT_DIR}"

uv run natural_language_create_model_prompts \
    model.pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    tokenization.output_format=yes_no \
    tokenization.enable_thinking=true \
    data_dir="${EHR_TEXT_DIR}" \
    output_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/thinking_yes_no"

uv run natural_language_create_model_prompts \
    model.pretrained_model_name_or_path="${PRETRAINED_MODEL_PATH}" \
    tokenization.output_format=probability \
    tokenization.enable_thinking=true \
    data_dir="${EHR_TEXT_DIR}" \
    output_dir="${NATURAL_LANGUAGE_PROMPT_DATA_DIR}/thinking_probability"
