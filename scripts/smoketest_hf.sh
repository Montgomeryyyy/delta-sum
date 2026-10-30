#!/usr/bin/env bash
# Smoke-test Hugging Face model load + summarize/delta prompts.
#
# Prerequisites:
#   - HF_TOKEN set (for gated models, e.g. Gemma)
#   - an already-synced project venv (do NOT use uv run — it pulls vllm)
#
# Usage:
#   source .env
#   source "$SCRATCH_DIR/.venv/bin/activate"   # or whatever your synced venv is
#   ./scripts/smoketest_hf.sh
#   ./scripts/smoketest_hf.sh --model google/gemma-3-270m --device cpu
#   ./scripts/smoketest_hf.sh --model Qwen/Qwen2.5-7B-Instruct --device cuda
#
# Optional sbatch (GPU):
#   sbatch --partition=slam_gpu --gres=gpu:1 --constraint=h100 \
#     --time=0:30:00 --cpus-per-task=8 --mem=32G --chdir="$(pwd)" \
#     --export=ALL,HF_TOKEN,VIRTUAL_ENV \
#     --output=smoketest_hf-%j.out \
#     scripts/smoketest_hf.sh --device cuda

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN must be set (e.g. source .env)." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

# Prefer an existing env — never `uv run` (that syncs and downloads the vllm wheel).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
    PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
elif [ -x "${SCRATCH_DIR:-}/.venv/bin/python" ]; then
    PYTHON="${SCRATCH_DIR}/.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    echo "Error: no project venv found. Activate one first, e.g.:" >&2
    echo "  source \"\$SCRATCH_DIR/.venv/bin/activate\"" >&2
    exit 1
fi

echo "Using Python: $PYTHON"
"$PYTHON" scripts/smoketest_hf.py "$@"
