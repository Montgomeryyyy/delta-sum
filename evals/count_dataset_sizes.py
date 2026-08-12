import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_from_disk


def get_splits(base_dir: Path):
    if not base_dir.exists():
        return {}
    # Ordered list of typical split names
    splits = {}
    for split_name in ["train", "tuning", "held_out"]:
        split_dir = base_dir / split_name
        if split_dir.exists():
            splits[split_name] = split_dir
    return splits


def count_parquet_rows(split_dir: Path):
    count = 0
    # recursive glob for parquets
    for p in split_dir.rglob("*.parquet"):
        try:
            count += pq.read_metadata(p).num_rows
        except Exception:
            pass
    return count


def count_tokens_in_hf(split_dir: Path):
    try:
        ds = load_from_disk(str(split_dir))
        table = ds.data

        # look for token column
        col_name = None
        for name in ["input_ids", "tokens"]:
            if name in table.column_names:
                col_name = name
                break

        if col_name:
            # this works for pyarrow ListArray
            try:
                # get lengths of all lists in the column
                lengths = pc.list_value_length(table.column(col_name))
                return pc.sum(lengths).as_py()
            except Exception:
                # fallback for datasets that might be formatted differently
                total = 0
                for batch in ds.iter(batch_size=1000):
                    for item in batch[col_name]:
                        total += len(item)
                return total
        else:
            return 0
    except Exception as e:
        print(f"Error counting tokens in {split_dir}: {e}")
        return 0


def count_patients_from_shards(shards: dict):
    counts = defaultdict(int)
    for key, patient_list in shards.items():
        # key format is usually 'train/0', 'tuning/0', etc.
        counts[key.split("/")[0]] += len(patient_list)

    return dict(counts)


def init_row(title: str):
    return {"Count": title, "train": 0, "tuning": 0, "held_out": 0}


def main():
    parser = argparse.ArgumentParser(description="Count dataset rows and tokens")
    parser.parse_args()

    # Determine base directories from env vars (assumes `source scripts/config.sh` has been run)
    try:
        meds_cohort_dir = os.environ["MIMICIV_MEDS_COHORT_DIR"]
        meds_tab_meds_dir = Path(os.environ["MEDS_TAB_MEDS_DIR"])
        nl_tokenized_dataset_dir = Path(os.environ["NATURAL_LANGUAGE_PRE_TRAINING_DIR"]) / "tokenized_dataset"
        struct_tokenized_dataset_dir = Path(os.environ["STRUCTURED_PRE_TRAINING_DIR"]) / "tokenized_dataset"
        aces_output_dir = Path(os.environ["ACES_OUTPUT_DIR"])
        eval_plots_dir = Path(os.environ["EVAL_PLOTS_DIR"])
    except KeyError as e:
        print(f"Error: Missing environment variable {e}. Did you forget to `source scripts/config.sh`?")
        return

    results = []

    # 0. Patients from MEDS_cohort
    shards_json_path = Path(meds_cohort_dir) / "metadata" / ".shards.json"
    if shards_json_path.exists():
        shards = json.loads(shards_json_path.read_text())
        patient_counts = count_patients_from_shards(shards)
        if patient_counts:
            row = init_row("Number of Patients")
            for split_name, cnt in patient_counts.items():
                if split_name in row:
                    row[split_name] = cnt
            results.append(row)

    # 1. MEDS_TAB_MEDS_DIR rows
    splits = get_splits(meds_tab_meds_dir)
    row = init_row("Number of Events")
    for split_name, split_dir in splits.items():
        if split_name in row:
            row[split_name] = count_parquet_rows(split_dir)
    results.append(row)

    # 2. Tokenized dataset tokens natural language
    splits = get_splits(nl_tokenized_dataset_dir)
    row = init_row("Number of Tokens (Natural Language)")
    for split_name, split_dir in splits.items():
        if split_name in row:
            row[split_name] = count_tokens_in_hf(split_dir)
    results.append(row)

    # 3. Tokenized dataset tokens structured
    splits = get_splits(struct_tokenized_dataset_dir)
    row = init_row("Number of Tokens (Structured)")
    for split_name, split_dir in splits.items():
        if split_name in row:
            row[split_name] = count_tokens_in_hf(split_dir)
    results.append(row)

    # 4. ACES_OUTPUT_DIR tasks rows
    if aces_output_dir.exists():
        for task_dir in sorted(aces_output_dir.iterdir()):
            if task_dir.is_dir():
                splits = get_splits(task_dir)
                if not splits:
                    continue
                row = init_row(f"Task examples {task_dir.name.replace('_', ' ').title()}")
                for split_name, split_dir in splits.items():
                    if split_name in row:
                        row[split_name] = count_parquet_rows(split_dir)
                results.append(row)

    df = pl.DataFrame(results)

    df = df.with_columns(total=pl.sum_horizontal("train", "tuning", "held_out"))

    # Configure polars printing
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(100)
    pl.Config.set_thousands_separator(",")

    print("\nDataset Sizes and Counts:")
    print(df)

    # Save to EVAL_PLOTS_DIR as markdown
    eval_plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_plots_dir / "dataset_sizes.md"

    with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True):
        out_path.write_text(str(df) + "\n")
    print(f"\nSaved markdown table to {out_path}")


if __name__ == "__main__":
    main()
