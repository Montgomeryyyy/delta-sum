"""Copy ACES labels for one task, keeping one label per subject per prediction date.

By default the earliest prediction time on each day is kept. The input labels are left untouched.

Example:
    uv run python mimic_struct_predict/preprocessing/one_label_per_day.py \
        --input_dir data/aces_labels/ptb --output_dir data/aces_labels_one_per_day/ptb
"""

import argparse
from pathlib import Path

import polars as pl


def one_label_per_day(df: pl.DataFrame, keep: str = "first") -> pl.DataFrame:
    return (
        df.sort("subject_id", "prediction_time")
        .with_columns(pl.col("prediction_time").dt.date().alias("_prediction_date"))
        .unique(subset=["subject_id", "_prediction_date"], keep=keep, maintain_order=True)
        .drop("_prediction_date")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Raw ACES output dir for one task.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keep", choices=["first", "last"], default="first")
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("--output_dir must differ from --input_dir so the raw labels are kept.")

    for path in sorted(args.input_dir.glob("*/*.parquet")):
        rel_path = path.relative_to(args.input_dir)
        df = pl.read_parquet(path)
        filtered = one_label_per_day(df, keep=args.keep)

        out_path = args.output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.write_parquet(out_path)
        print(f"{rel_path}: {df.height:,} -> {filtered.height:,} rows")


if __name__ == "__main__":
    main()
