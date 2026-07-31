import argparse
from pathlib import Path

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--debug_dir", type=str, required=True)
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    debug_dir = Path(args.debug_dir)

    if not task_dir.exists():
        print(f"Output directory does not exist: {task_dir}")
        return

    # Iterate over all task directories
    task_dirs = [d for d in task_dir.iterdir() if d.is_dir()]

    for task_dir in task_dirs:
        task_name = task_dir.name
        train_dir = task_dir / "train"

        if not train_dir.exists():
            continue

        parquet_files = list(train_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        # Read the first parquet file for the task
        df = pl.read_parquet(parquet_files[0])

        if "ehr_text" not in df.columns:
            print(f"Column 'ehr_text' not found in task {task_name}, skipping.")
            continue

        # Determine subjects
        if "subject_id" in df.columns:
            subjects = df["subject_id"].unique().head(10).to_list()
        else:
            subjects = list(range(min(10, len(df))))

        task_debug_dir = debug_dir / task_name
        task_debug_dir.mkdir(parents=True, exist_ok=True)

        for idx, subject_id in enumerate(subjects):
            subject_df = df.filter(pl.col("subject_id") == subject_id)
            output_file = task_debug_dir / f"{subject_id}.md"
            output_file.write_text(subject_df["ehr_text"][0], encoding="utf-8")


if __name__ == "__main__":
    main()
