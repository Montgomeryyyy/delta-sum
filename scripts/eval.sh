#!/usr/bin/env bash

# Usage:
#   ./scripts/eval_plots.sh <NL_TOKENIZER_PATH> <STRUCT_TOKENIZER_PATH>
#   ./scripts/eval_plots.sh --nl_tokenizer_path <PATH> --struct_tokenizer_path <PATH>

set -e

NL_TOKENIZER_PATH=""
STRUCT_TOKENIZER_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nl_tokenizer_path|--nl-tokenizer-path)
      NL_TOKENIZER_PATH="$2"
      shift 2
      ;;
    --struct_tokenizer_path|--struct-tokenizer-path)
      STRUCT_TOKENIZER_PATH="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 <nl_tokenizer_path> <struct_tokenizer_path>"
      exit 0
      ;;
    *)
      if [ -z "$NL_TOKENIZER_PATH" ]; then
        NL_TOKENIZER_PATH="$1"
        shift
      elif [ -z "$STRUCT_TOKENIZER_PATH" ]; then
        STRUCT_TOKENIZER_PATH="$1"
        shift
      else
        echo "Unknown option: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [ -z "$NL_TOKENIZER_PATH" ] || [ -z "$STRUCT_TOKENIZER_PATH" ]; then
    echo "Usage: $0 <nl_tokenizer_path> <struct_tokenizer_path>" >&2
    exit 1
fi

# --- Get config ---
source "./scripts/config.sh"

echo "---- Dataset Count ----"
uv run python mimic_struct_predict/evals/dataset_count.py

echo "---- Error Rates ----"
uv run python mimic_struct_predict/evals/error_rates.py

echo "---- Event Config to LaTeX ----"
uv run python mimic_struct_predict/evals/event_config_to_latex.py

echo "---- Rollout Delta AUROC ----"
uv run python mimic_struct_predict/evals/rollout_delta.py --metric auroc

echo "---- Rollout Delta Brier Score ----"
uv run python mimic_struct_predict/evals/rollout_delta.py --metric brier_score

echo "---- Training Details ----"
uv run python mimic_struct_predict/evals/training_details.py

echo "---- GT Trajectory Lengths ----"
uv run python mimic_struct_predict/evals/gt_trajectory_lengths.py --nl_tokenizer_path "$NL_TOKENIZER_PATH" --struct_tokenizer_path "$STRUCT_TOKENIZER_PATH"
