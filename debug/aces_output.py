import argparse
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--aces_train_dir", type=str, required=True)
    parser.add_argument("--meds_train_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aces_df = pl.read_parquet(args.aces_train_dir)
    meds_df = pl.read_parquet(args.meds_train_dir)

    subjects = aces_df["subject_id"].unique().head(10)

    for subject_id in subjects.to_list():
        subject_aces = aces_df.filter(pl.col("subject_id") == subject_id)
        if len(subject_aces) == 0:
            continue

        prediction_time = subject_aces["prediction_time"][0]
        boolean_value = subject_aces["boolean_value"][0]

        subject_meds = meds_df.filter(pl.col("subject_id") == subject_id)

        input_meds = subject_meds.filter(pl.col("time") <= prediction_time)
        output_meds = subject_meds.filter(pl.col("time") > prediction_time)

        input_csv = input_meds.to_pandas().to_csv(index=False)
        output_csv = output_meds.to_pandas().to_csv(index=False)

        out_str = f"boolean_value: {boolean_value}\n=== INPUT ===\n{input_csv}\n=== OUTPUT ===\n{output_csv}"

        output_file = output_dir / f"{subject_id}.csv"
        output_file.write_text(out_str)


if __name__ == "__main__":
    main()
