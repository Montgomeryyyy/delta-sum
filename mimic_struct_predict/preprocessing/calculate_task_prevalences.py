"""Calculate task prevalences from ACES training dataset shards."""

import argparse
import json
import logging
import os
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def calculate_task_prevalence(task_split_dir: Path) -> tuple[int, int, float]:
    """Calculate total rows, positive rows, and prevalence for a single task split directory."""
    if not task_split_dir.exists() or not task_split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {task_split_dir}")

    parquet_files = sorted(task_split_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {task_split_dir}")

    total_rows = 0
    total_positives = 0

    for file_path in parquet_files:
        df = pl.read_parquet(file_path, columns=["boolean_value"])
        total_rows += len(df)
        if "boolean_value" in df.columns:
            total_positives += int(df["boolean_value"].cast(pl.Int64).fill_null(0).sum())

    if total_rows == 0:
        raise ValueError(f"No rows found in parquet files under {task_split_dir}")

    prevalence = total_positives / total_rows
    return total_rows, total_positives, prevalence


def compute_all_task_prevalences(
    aces_dir: Path,
    split: str = "train",
    tasks: list[str] | None = None,
) -> dict[str, float]:
    """Compute prevalences for all tasks in ACES directory for the specified split."""
    if not aces_dir.exists() or not aces_dir.is_dir():
        raise FileNotFoundError(f"ACES directory not found: {aces_dir}")

    if tasks is not None and len(tasks) > 0:
        task_names = sorted(tasks)
    else:
        task_names = sorted([d.name for d in aces_dir.iterdir() if d.is_dir() and (d / split).exists()])

    if not task_names:
        raise ValueError(f"No valid task directories with split '{split}' found in {aces_dir}")

    prevalences: dict[str, float] = {}
    summary_rows = []

    for task_name in task_names:
        split_dir = aces_dir / task_name / split
        total_rows, total_positives, prevalence = calculate_task_prevalence(split_dir)
        prevalences[task_name] = prevalence
        summary_rows.append(
            {
                "Task": task_name,
                "Split": split,
                "Total Rows": total_rows,
                "Positive Rows": total_positives,
                "Prevalence": prevalence,
            }
        )

    summary_df = pl.DataFrame(summary_rows)
    log.info(f"\nTask Prevalences Summary ({split} split):\n{summary_df}")

    return prevalences


def save_task_prevalences(prevalences: dict[str, float], output_path: Path) -> None:
    """Save task prevalences dictionary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prevalences, indent=2), encoding="utf-8")
    log.info(f"Saved task prevalences to {output_path}")


def main() -> None:
    """CLI entry point for calculating task prevalences."""
    default_aces_dir = os.environ.get("ACES_OUTPUT_DIR")
    default_output_path = Path(default_aces_dir) / "task_prevalences.json" if default_aces_dir else None

    parser = argparse.ArgumentParser(
        description="Calculate task prevalences from ACES training dataset shards and export to JSON."
    )
    parser.add_argument(
        "--aces-dir",
        type=Path,
        default=Path(default_aces_dir) if default_aces_dir else None,
        required=default_aces_dir is None,
        help="Path to the ACES output directory containing task folders. Defaults to $ACES_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path,
        required=default_output_path is None,
        help="Path where the output JSON file will be written. Defaults to $ACES_OUTPUT_DIR/task_prevalences.json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to compute prevalence from (default: 'train').",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of specific task names to process. If omitted, all tasks found in aces-dir are processed.",
    )

    args = parser.parse_args()

    log.info(f"Computing task prevalences for split '{args.split}' from: {args.aces_dir}")
    prevalences = compute_all_task_prevalences(
        aces_dir=args.aces_dir,
        split=args.split,
        tasks=args.tasks,
    )

    save_task_prevalences(prevalences, args.output_path)


if __name__ == "__main__":
    main()
