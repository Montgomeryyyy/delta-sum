import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--meds_train_dir", type=str)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meds_train_dir = args.meds_train_dir

    df = pl.read_parquet(meds_train_dir)
    subjects = df["subject_id"].unique().sort().head(10)

    for subject_id in subjects.to_list():
        subject_df = df.filter(pl.col("subject_id") == subject_id)
        csv_str = subject_df.to_pandas().to_csv(index=False)
        output_file = output_dir / f"{subject_id}.csv"
        output_file.write_text(csv_str)


if __name__ == "__main__":
    main()
