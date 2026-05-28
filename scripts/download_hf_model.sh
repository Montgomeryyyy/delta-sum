#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

# Print commands
set -x

# --- Get download models ----
uv run hf download $HF_MODEL_NAME --local-dir $PRETRAINED_MODEL_PATH
