#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Get config ---
source "./scripts/config.sh"

export SERVER_PORT=8000

echo "Run this command locally to port forward: ssh -NL ${SERVER_PORT}:localhost:${SERVER_PORT} ${HOSTNAME}"

# uv run tensorboard --port=${SERVER_PORT} --logdir ${SCRATCH_DIR}/training/natural_language/grpo/thinking_yes_no
uv run tensorboard --port=${SERVER_PORT} --logdir ${SCRATCH_DIR}/training/natural_language/grpo/
# uv run tensorboard --port=${SERVER_PORT} --logdir ${SCRATCH_DIR}/training/natural_language/sft/
# uv run tensorboard --port=${SERVER_PORT} --logdir ${SCRATCH_DIR}/training/natural_language/grpo/direct_yes_no
