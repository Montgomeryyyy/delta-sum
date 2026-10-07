#!/usr/bin/env bash

# Usage:
#   ./scripts/pretrain-dataset.sh [MODE]
#
# Arguments:
#   MODE: Either 'natural_language' (default) or 'structured'.
#         Determines which configuration and output directory are used.

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Pipeline Execution ---

# Print commands
# set -x

SUM_MODE=${1}

if [[ "$SUM_MODE" == "all" ]]; then
    CONFIG_NAME="summarize.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_RAW_DATA_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_SUMMARIZED_DATA_DIR}"
else if [[ "$SUM_MODE" == "delta_summarize" ]]; then
    CONFIG_NAME="delta_summarize.yaml"
    DATASET_DIR="${NATURAL_LANGUAGE_RAW_DATA_DIR}"
    OUTPUT_DIR="${NATURAL_LANGUAGE_DELTA_SUMMARIZED_DATA_DIR}"
else
    echo "Invalid summary mode: $SUM_MODE"
    exit 1
fi