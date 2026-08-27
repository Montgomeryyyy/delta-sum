#!/usr/bin/env bash

set -e

# --- Get config ---
source "./scripts/config.sh"

uv run python evals/rollout_delta.py "$@"
