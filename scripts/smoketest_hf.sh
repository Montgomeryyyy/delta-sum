#!/usr/bin/env bash
# Smoke-test Hugging Face model load + summarize/delta prompts.
#
# Prerequisites:
#   - HF_TOKEN set (for gated models, e.g. Gemma)
#   - an env with torch + transformers already installed
#
# Usage:
#   source .env
#   conda activate delta_sum   # or activate a synced uv venv
#   ./scripts/smoketest_hf.sh
#   ./scripts/smoketest_hf.sh --model google/gemma-3-270m --device cpu
#   ./scripts/smoketest_hf.sh --model Qwen/Qwen2.5-7B-Instruct --device cuda

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN must be set (e.g. source .env)." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

# Prefer the *active* env. Conda sets CONDA_PREFIX, not VIRTUAL_ENV.
# Never `uv run` here (that syncs and downloads the vllm wheel).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
    PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
elif [ -x "${SCRATCH_DIR:-}/.venv/bin/python" ]; then
    PYTHON="${SCRATCH_DIR}/.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python || true)"
fi

if [ -z "${PYTHON:-}" ] || [ ! -x "$PYTHON" ]; then
    echo "Error: no Python found. Activate an env with torch installed." >&2
    exit 1
fi

echo "Using Python: $PYTHON"
if ! "$PYTHON" -c "import torch, transformers" 2>/dev/null; then
    echo "Error: $PYTHON is missing torch/transformers." >&2
    echo "Install them in your active env, e.g.:" >&2
    echo "  pip install torch transformers" >&2
    exit 1
fi

"$PYTHON" scripts/smoketest_hf.py "$@"
