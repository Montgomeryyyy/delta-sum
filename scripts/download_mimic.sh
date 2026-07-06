#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

DO_DEMO="false"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --demo) DO_DEMO="true"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# --- Get config ---
source "./scripts/config.sh"

# --- Credentials Check ---
if [ -z "$MIMIC_USER" ] || [ -z "$MIMIC_PASSWORD" ]; then
    echo "Error: MIMIC_USER and MIMIC_PASSWORD environment variables must be set." >&2
    exit 1
fi

DATASET_NAME="mimiciv"
MIMICIV_RAW_DIR="${DATA_DIR}/raw_mimic"

# --- Main Download Process ---
mkdir -p "$MIMICIV_RAW_DIR"

# Download main MIMIC-IV demo dataset
rm -rf ${MIMICIV_RAW_DIR}
echo "Starting download for ${DATASET_NAME}..."
uv run physionet_download \
    raw_mimic_dir="${MIMICIV_RAW_DIR}" \
    mimic_version="${MIMIC_VERSION}" \
    do_demo="${DO_DEMO}"
