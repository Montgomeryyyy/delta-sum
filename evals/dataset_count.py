import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_from_disk

from evals.plot_style import TASK_MAP, format_task_title, sort_tasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SPLITS = ["train", "tuning", "held_out"]


def count_patients(shards_json_path: Path) -> dict[str, int]:
    """Counts unique patients per split from .shards.json metadata."""
    shards = json.loads(shards_json_path.read_text())
    counts = defaultdict(int)
    for key, patient_list in shards.items():
        # Format train/0 etc...
        counts[key.split("/")[0]] += len(patient_list)
    return dict(counts)


def count_parquet_rows(split_dir: Path) -> int:
    """Counts rows across all parquet files in a directory."""
    return sum(pq.read_metadata(p).num_rows for p in split_dir.rglob("*.parquet"))


def count_tokens(split_dir: Path, col_name="input_ids") -> int:
    """Counts total tokens in a Hugging Face tokenized dataset split. Assumes no padding has been added"""
    if not split_dir.exists():
        return 0
    ds = load_from_disk(str(split_dir))
    table = ds.data
    lengths = pc.list_value_length(table.column(col_name))
    return pc.sum(lengths).as_py()


def count_task_stats(task_dir: Path) -> dict[str, tuple[int, int]]:
    """Computes total rows and positive rows per split for an ACES task."""
    stats = {}
    for split in SPLITS:
        files = sorted((task_dir / split).glob("*.parquet"))
        res = pl.scan_parquet(files).select(total=pl.len(), pos=pl.col("boolean_value").cast(pl.Int64).sum()).collect()
        stats[split] = (res["total"][0], res["pos"][0])
    return stats


def format_task_cell(count: int, pos: int) -> str:
    """Formats task cell with count and prevalence percentage in brackets."""
    prev = (pos / count * 100) if count > 0 else 0.0
    return f"{count:,} ({prev:.1f}%)"


def save_latex_table(df: pl.DataFrame, out_path: Path) -> None:
    """Generates a camera-ready booktabs LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Train} & \textbf{Tuning} & \textbf{Held-Out} & \textbf{Total} \\",
        r"\midrule",
    ]

    task_titles = {format_task_title(t) for t in TASK_MAP}
    task_section_started = False
    for row in df.iter_rows(named=True):
        title = row["Metric"]
        if title in task_titles and not task_section_started:
            lines.append(r"\midrule")
            task_section_started = True

        cells = [
            title,
            row["Train"].replace("%", r"\%"),
            row["Tuning"].replace("%", r"\%"),
            row["Held-Out"].replace("%", r"\%"),
            row["Total"].replace("%", r"\%"),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Overview of dataset sizes, split counts, and task prevalences.}",
            r"\label{tab:dataset_sizes}",
            r"\end{table}",
            "",
        ]
    )

    out_path.write_text("\n".join(lines))
    log.info(f"Saved LaTeX table to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Metric dataset rows, tokens, and task prevalences.")
    parser.add_argument(
        "--meds_cohort_dir",
        type=Path,
        default=Path(os.environ["MIMICIV_MEDS_COHORT_DIR"]),
        help="Path to MIMICIV MEDS cohort directory",
    )
    parser.add_argument(
        "--meds_tab_meds_dir",
        type=Path,
        default=Path(os.environ["MEDS_TAB_MEDS_DIR"]),
        help="Path to MEDS_TAB merged cohort directory",
    )
    parser.add_argument(
        "--nl_tokenized_dataset_dir",
        type=Path,
        default=Path(os.environ["NATURAL_LANGUAGE_PRE_TRAINING_DIR"]) / "tokenized_dataset",
        help="Path to natural language tokenized dataset directory",
    )
    parser.add_argument(
        "--struct_tokenized_dataset_dir",
        type=Path,
        default=Path(os.environ["STRUCTURED_PRE_TRAINING_DIR"]) / "tokenized_dataset",
        help="Path to structured tokenized dataset directory",
    )
    parser.add_argument(
        "--aces_dir",
        type=Path,
        default=Path(os.environ["ACES_OUTPUT_DIR"]),
        help="Path to ACES labels directory",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=os.environ.get("ACES_TASKS", "").split(),
        help="List of task names to count",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ["EVAL_PLOTS_DIR"]) / "dataset_count",
        help="Output directory for CSV, LaTeX, and Markdown tables",
    )
    args = parser.parse_args()

    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    rows = []

    # 1. Number of patients
    shards_json = args.meds_cohort_dir / "metadata" / ".shards.json"
    patient_counts = count_patients(shards_json)
    train_p = patient_counts["train"]
    tuning_p = patient_counts["tuning"]
    held_out_p = patient_counts["held_out"]
    rows.append(
        {
            "Metric": "Patients",
            "Train": f"{train_p:,}",
            "Tuning": f"{tuning_p:,}",
            "Held-Out": f"{held_out_p:,}",
            "Total": f"{train_p + tuning_p + held_out_p:,}",
        }
    )

    # 2. Number of events
    train_events = count_parquet_rows(args.meds_tab_meds_dir / "train")
    tuning_events = count_parquet_rows(args.meds_tab_meds_dir / "tuning")
    held_out_events = count_parquet_rows(args.meds_tab_meds_dir / "held_out")
    rows.append(
        {
            "Metric": "Events",
            "Train": f"{train_events:,}",
            "Tuning": f"{tuning_events:,}",
            "Held-Out": f"{held_out_events:,}",
            "Total": f"{train_events + tuning_events + held_out_events:,}",
        }
    )

    # 3. Pre-training tokens (Natural Language)
    train_nl = count_tokens(args.nl_tokenized_dataset_dir / "train")
    tuning_nl = count_tokens(args.nl_tokenized_dataset_dir / "tuning")
    held_out_nl = count_tokens(args.nl_tokenized_dataset_dir / "held_out")
    rows.append(
        {
            "Metric": "Pre-training Tokens (Natural Language)",
            "Train": f"{train_nl:,}",
            "Tuning": f"{tuning_nl:,}",
            "Held-Out": f"{held_out_nl:,}",
            "Total": f"{train_nl + tuning_nl + held_out_nl:,}",
        }
    )

    # 4. Pre-training tokens (Structured)
    train_struct = count_tokens(args.struct_tokenized_dataset_dir / "train")
    tuning_struct = count_tokens(args.struct_tokenized_dataset_dir / "tuning")
    held_out_struct = count_tokens(args.struct_tokenized_dataset_dir / "held_out")
    rows.append(
        {
            "Metric": "Pre-training Tokens (Structured)",
            "Train": f"{train_struct:,}",
            "Tuning": f"{tuning_struct:,}",
            "Held-Out": f"{held_out_struct:,}",
            "Total": f"{train_struct + tuning_struct + held_out_struct:,}",
        }
    )

    # 5. Task examples & prevalences
    for task_name in args.tasks:
        stats = count_task_stats(args.aces_dir / task_name)
        train_tot, train_pos = stats["train"]
        tuning_tot, tuning_pos = stats["tuning"]
        held_out_tot, held_out_pos = stats["held_out"]
        total_tot = train_tot + tuning_tot + held_out_tot
        total_pos = train_pos + tuning_pos + held_out_pos

        rows.append(
            {
                "Metric": format_task_title(task_name),
                "Train": format_task_cell(train_tot, train_pos),
                "Tuning": format_task_cell(tuning_tot, tuning_pos),
                "Held-Out": format_task_cell(held_out_tot, held_out_pos),
                "Total": format_task_cell(total_tot, total_pos),
            }
        )

    df = pl.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dataset_sizes.csv"
    tex_path = args.output_dir / "dataset_sizes.tex"
    md_path = args.output_dir / "dataset_sizes.md"

    df.write_csv(csv_path)
    log.info(f"Saved CSV table to {csv_path}")

    save_latex_table(df, tex_path)

    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        md_path.write_text(str(df) + "\n")
    log.info(f"Saved Markdown table to {md_path}")

    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=100):
        print("\nDataset Sizes and Counts:")
        print(df)


if __name__ == "__main__":
    main()
