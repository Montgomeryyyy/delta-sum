#!/usr/bin/env bash
# Smoke-test Hugging Face model load + summarize/delta prompts.
#
# Prerequisites (same as the rest of this repo):
#   - SCRATCH_DIR and HF_TOKEN set (e.g. via .env)
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
#     --export=ALL,SCRATCH_DIR,HF_TOKEN \
#     --output="$SCRATCH_DIR/../logs/smoketest_hf/%x-%j.out" \
#     scripts/smoketest_hf.sh --device cuda

set -euo pipefail

source "./scripts/config.sh"

# Ensure HF hub auth for gated models (e.g. Gemma).
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"

uv run python scripts/smoketest_hf.py "$@"
