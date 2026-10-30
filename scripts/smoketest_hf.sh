#!/usr/bin/env bash
# Smoke-test Hugging Face model load + summarize/delta prompts.
#
# Prerequisites:
#   - HF_TOKEN set (for gated models, e.g. Gemma)
#   - project deps installed:  uv sync
#
# Usage:
#   ./scripts/smoketest_hf.sh
#   ./scripts/smoketest_hf.sh --model google/gemma-3-270m --device cpu
#   ./scripts/smoketest_hf.sh --model Qwen/Qwen2.5-7B-Instruct --device cuda
#
# Optional sbatch (GPU):
#   sbatch --partition=slam_gpu --gres=gpu:1 --constraint=h100 \
#     --time=0:30:00 --cpus-per-task=8 --mem=32G --chdir="$(pwd)" \
#     --export=ALL,HF_TOKEN \
#     --output=smoketest_hf-%j.out \
#     scripts/smoketest_hf.sh --device cuda

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN must be set (e.g. source .env)." >&2
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

uv run python scripts/smoketest_hf.py "$@"
