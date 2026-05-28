#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# --- Credentials Check ---
if [ -z "$MIMIC_USER" ] || [ -z "$MIMIC_PASSWORD" ]; then
    echo "Error: MIMIC_USER and MIMIC_PASSWORD environment variables must be set." >&2
    exit 1
fi

# --- Main Download Process ---
mkdir -p "$MIMICIV_RAW_DIR"

# Download main MIMIC-IV demo dataset
echo "Starting download for ${DATASET_NAME}..."
cd "$MIMICIV_RAW_DIR" && \
    wget -q -r -N -c -np --user "$MIMIC_USER" --password "$MIMIC_PASSWORD" \
        --no-check-certificate --accept gz,csv --reject "index.html*" -nH --cut-dirs=3 \
        "${PHYSIONET_URL}/${DATASET_NAME}/${MIMIC_VERSION}/"
echo "Downloaded ${DATASET_NAME} v${MIMIC_VERSION} to ${MIMICIV_RAW_DIR}"

# Download MIMIC-IV ED demo dataset
echo "Starting download for ${DATASET_NAME_ED}..."
wget -q -r -N -c -np --user "$MIMIC_USER" --password "$MIMIC_PASSWORD" \
    --no-check-certificate --accept gz,csv --reject "index.html*" -nH --cut-dirs=3 \
    "${PHYSIONET_URL}/${DATASET_NAME_ED}/${MIMIC_VERSION}/"
echo "Downloaded ${DATASET_NAME_ED} v${MIMIC_VERSION} to ${MIMICIV_RAW_DIR}"
