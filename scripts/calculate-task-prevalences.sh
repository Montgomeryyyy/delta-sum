#!/usr/bin/env bash

# Calculate task prevalences from ACES training dataset shards.
#
# Usage:
#   ./scripts/calculate-task-prevalences.sh [--aces-dir DIR] [--output-path PATH] [--split SPLIT] [--tasks TASK ...]

set -e

# --- Get config ---
source "./scripts/config.sh"

uv run calculate_task_prevalences "$@"
