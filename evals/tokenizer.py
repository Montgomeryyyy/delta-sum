import collections
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from datasets import disable_caching, disable_progress_bar, load_dataset
from transformers import AutoTokenizer

disable_progress_bar()
disable_caching()

# Camera-ready plot settings
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Setup config from environment variables (assuming `source scripts/config.sh` was run)
LINEWIDTH = 5.5
CONTEXT_WINDOW = 4096
NL_COLOR = "#1f77b4"
STRUCT_COLOR = "#ff7f0e"
UNTRUNCATED_COLOR = "#2ca02c"
HF_MODEL_NAME = os.environ["HF_MODEL_NAME"]
NUM_WORKERS = int(os.environ["NUM_WORKERS"])
EVAL_PLOTS_DIR = Path(os.environ["EVAL_PLOTS_DIR"])

NL_PRE_TRAINING_DIR = Path(os.environ["NATURAL_LANGUAGE_PRE_TRAINING_DIR"])
STRUCT_PRE_TRAINING_DIR = Path(os.environ["STRUCTURED_PRE_TRAINING_DIR"])
HEADER_MEDS_DIR = Path(os.environ["MIMICIV_MEDS_COHORT_DIR"]) / "order_by_code_prefix"
MEDS_TAB_MEDS_DIR = Path(os.environ["MEDS_TAB_MEDS_DIR"])

EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Compile regex patterns once globally
DATE_PATTERN = re.compile(r"## Date.*?\+(?P<days>\d+)\s*days")
TIME_PATTERN = re.compile(r"### Time.*?(?P<hours>\d+)h.*?(?P<minutes>\d+)m")


def plot_header_counts():
    print("Calculating header title counts...")
    header_counter = collections.Counter()
    for file in HEADER_MEDS_DIR.rglob("*.parquet"):
        df = pl.read_parquet(file, columns=["code"])
        headers = df.get_column("code").str.split("//").list.first().str.split(":").list.first().drop_nulls()
        header_counter.update(headers.to_list())

    print("Plotting header title counts...")
    header_df = pd.DataFrame.from_dict(header_counter, orient="index", columns=["Count"]).reset_index()
    header_df.columns = ["Event Type", "Event Count"]
    header_df = header_df.sort_values(by="Event Count", ascending=False)

    plt.figure(figsize=(LINEWIDTH, 4.5))
    sns.barplot(x="Event Count", y="Event Type", data=header_df, color=NL_COLOR)
    plt.xscale("log")
    plt.xlabel("Event Count")
    plt.ylabel("")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    header_plot_path = EVAL_PLOTS_DIR / "header_counts.pdf"
    plt.savefig(header_plot_path)
    plt.close()
    print(f"Saved Header Counts plot to {header_plot_path}")


def get_length(example, tokenizer):
    return {"length": len(tokenizer.encode(example["ehr_text"], add_special_tokens=True))}


def plot_cdf_sequence_lengths(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer):
    print("Calculating raw token lengths...")

    # Using map with num_proc for faster parallel execution across the whole dataset
    nl_raw = nl_raw.map(get_length, fn_kwargs={"tokenizer": nl_tokenizer}, num_proc=NUM_WORKERS)
    struct_raw = struct_raw.map(get_length, fn_kwargs={"tokenizer": struct_tokenizer}, num_proc=NUM_WORKERS)

    nl_lengths = np.array(nl_raw["length"])
    struct_lengths = np.array(struct_raw["length"])

    print("Plotting CDF of sequence lengths...")
    plt.figure(figsize=(LINEWIDTH, 3.5))

    # Calculate and plot CDF for NL
    nl_sorted = np.sort(nl_lengths)
    nl_cdf = np.arange(1, len(nl_sorted) + 1) / len(nl_sorted)
    nl_sorted = np.concatenate(([0], nl_sorted))
    nl_cdf = np.concatenate(([0], nl_cdf))
    plt.plot(nl_sorted, nl_cdf, label="Natural Language", color=NL_COLOR)

    # Calculate and plot CDF for Struct
    struct_sorted = np.sort(struct_lengths)
    struct_cdf = np.arange(1, len(struct_sorted) + 1) / len(struct_sorted)
    struct_sorted = np.concatenate(([0], struct_sorted))
    struct_cdf = np.concatenate(([0], struct_cdf))
    plt.plot(struct_sorted, struct_cdf, label="Structured", color=STRUCT_COLOR)

    # Intercept
    plt.axvline(x=CONTEXT_WINDOW, color="red", linestyle="--", label=f"{CONTEXT_WINDOW} Context Window")

    # Set x-lim up to 99th percentile to avoid huge outliers skewing the plot
    max_x = max(np.percentile(nl_lengths, 99), np.percentile(struct_lengths, 99))
    plt.xscale("log", base=2)
    # HACK: Hardcoded cut-off point
    min_log2_x = 7
    plt.xlim(2**min_log2_x, max_x)
    ticks = [2**i for i in range(min_log2_x, int(np.ceil(np.log2(max_x))) + 1)]
    plt.xticks(ticks, [f"$2^{{{int(np.log2(t))}}}$" for t in ticks])
    plt.ylim(0, 1)
    plt.xlabel("Full Tokenized EHR Length")
    plt.ylabel("Cumulative Fraction\nof Sequences")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    cdf_plot_path = EVAL_PLOTS_DIR / "cdf_plot.pdf"
    plt.savefig(cdf_plot_path)
    plt.close()
    print(f"Saved CDF plot to {cdf_plot_path}")

    nl_intercept_pct = np.mean(nl_lengths <= CONTEXT_WINDOW)
    struct_intercept_pct = np.mean(struct_lengths <= CONTEXT_WINDOW)
    print(f"NL CDF at {CONTEXT_WINDOW}: {nl_intercept_pct:.4f}")
    print(f"Struct CDF at {CONTEXT_WINDOW}: {struct_intercept_pct:.4f}")


def extract_metrics(input_ids, tokenizer, split_pattern):
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    events = 0
    current_day = 0
    start_time = None
    end_time = None

    for line in text.split(split_pattern):
        line = line.strip()
        if not line:
            continue

        if not line.startswith("#"):
            events += 1
            continue

        date_match = DATE_PATTERN.search(line)
        if date_match:
            current_day += int(date_match.group("days"))
            continue

        time_match = TIME_PATTERN.search(line)
        if time_match:
            hours = int(time_match.group("hours"))
            minutes = int(time_match.group("minutes"))
            absolute_time = current_day + hours / 24.0 + minutes / (24.0 * 60.0)

            if start_time is None:
                start_time = absolute_time
            else:
                end_time = absolute_time
        elif "### Time Unspecified" in line:  # Unspecified is set to EOD
            absolute_time = current_day + 23 / 24.0 + 59 / (24.0 * 60.0)

            if start_time is None:
                start_time = absolute_time
            else:
                end_time = absolute_time

    if start_time is None:
        raise ValueError(f"No start time found in EHR:\n{text}")
    elif end_time is None:
        time_covered = 0.0
    else:
        time_covered = end_time - start_time
        if time_covered < 0:
            raise ValueError(f"Negative time found EHR:\n{text}")

    return {"events": events, "time_covered": time_covered}


def plot_events_and_timeline_boxplots(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer):
    def tokenize_and_truncate(examples, tokenizer):
        return tokenizer(examples["ehr_text"], truncation=True, max_length=CONTEXT_WINDOW, add_special_tokens=True)

    print("Tokenizing datasets with truncation...")
    nl_tok_ds = nl_raw.map(
        tokenize_and_truncate,
        fn_kwargs={"tokenizer": nl_tokenizer},
        batched=True,
        num_proc=NUM_WORKERS,
        remove_columns=nl_raw.column_names,
    )
    struct_tok_ds = struct_raw.map(
        tokenize_and_truncate,
        fn_kwargs={"tokenizer": struct_tokenizer},
        batched=True,
        num_proc=NUM_WORKERS,
        remove_columns=struct_raw.column_names,
    )

    print("Extracting metrics...")
    nl_tok_ds = nl_tok_ds.map(
        extract_metrics,
        fn_kwargs={"tokenizer": nl_tokenizer, "split_pattern": "\n\n"},
        input_columns="input_ids",
        num_proc=NUM_WORKERS,
    )
    # It is \n\n here not //\n\n// because the tokenizer doesn't readd the // on decode (maybe it should...)
    struct_tok_ds = struct_tok_ds.map(
        extract_metrics,
        fn_kwargs={"tokenizer": struct_tokenizer, "split_pattern": "\n\n"},
        input_columns="input_ids",
        num_proc=NUM_WORKERS,
    )

    nl_event_counts = np.array(nl_tok_ds["events"])
    struct_event_counts = np.array(struct_tok_ds["events"])

    print("Loading Complete MEDS dataset...")
    complete_meds = (
        pl.scan_parquet(str(MEDS_TAB_MEDS_DIR / "train" / "*.parquet"))
        .group_by("subject_id")
        .agg([pl.len().alias("events"), (pl.col("time").max() - pl.col("time").min()).alias("time_covered")])
        .collect()
    )
    complete_event_counts = complete_meds["events"].to_numpy()
    complete_time_covered_vals = complete_meds["time_covered"].dt.total_milliseconds().to_numpy() / (
        1000 * 60 * 60 * 24
    )

    print("Plotting Boxplot for events...")
    plt.figure(figsize=(LINEWIDTH, 4))

    df = pd.DataFrame(
        {
            "Events": np.concatenate([nl_event_counts, struct_event_counts, complete_event_counts]),
            "Dataset": [f"Natural Language\nContext Window={CONTEXT_WINDOW}"] * len(nl_event_counts)
            + [f"Structured\nContext Window={CONTEXT_WINDOW}"] * len(struct_event_counts)
            + ["Untruncated"] * len(complete_event_counts),
        }
    )

    palette = {
        f"Natural Language\nContext Window={CONTEXT_WINDOW}": NL_COLOR,
        f"Structured\nContext Window={CONTEXT_WINDOW}": STRUCT_COLOR,
        "Untruncated": UNTRUNCATED_COLOR,
    }
    sns.boxplot(
        x="Dataset", y="Events", data=df, hue="Dataset", palette=palette, showfliers=False, width=0.5, legend=False
    )

    plt.ylabel("Events in Timeline")
    plt.xlabel("")
    plt.ylim(bottom=0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    events_plot_path = EVAL_PLOTS_DIR / "events_boxplot.pdf"
    plt.savefig(events_plot_path)
    plt.close()
    print(f"Saved Events Boxplot to {events_plot_path}")

    nl_time_covered_vals = np.array(nl_tok_ds["time_covered"])
    struct_time_covered_vals = np.array(struct_tok_ds["time_covered"])

    print("Plotting Boxplot for timeline length...")
    plt.figure(figsize=(LINEWIDTH, 4))

    df_time = pd.DataFrame(
        {
            "Days": np.concatenate([nl_time_covered_vals, struct_time_covered_vals, complete_time_covered_vals]),
            "Dataset": [f"Natural Language\nContext Window={CONTEXT_WINDOW}"] * len(nl_time_covered_vals)
            + [f"Structured\nContext Window={CONTEXT_WINDOW}"] * len(struct_time_covered_vals)
            + ["Untruncated"] * len(complete_time_covered_vals),
        }
    )

    sns.boxplot(
        x="Dataset", y="Days", data=df_time, hue="Dataset", palette=palette, showfliers=False, width=0.5, legend=False
    )

    plt.ylabel("Days in Timeline")
    plt.xlabel("")
    plt.ylim(bottom=0)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    days_plot_path = EVAL_PLOTS_DIR / "days_boxplot.pdf"
    plt.savefig(days_plot_path)
    plt.close()
    print(f"Saved Days Boxplot to {days_plot_path}")


def extract_trajectory(example, tokenizer, split_pattern="\n\n"):
    text = example["ehr_text"]
    lines = text.split(split_pattern)

    # Batch tokenize all lines at once for massive speedup
    lines_with_split = [line + split_pattern for line in lines]
    encoded_lines = tokenizer(lines_with_split, add_special_tokens=False)["input_ids"]
    line_token_lengths = [len(x) for x in encoded_lines]

    event_tokens = []

    day_x = []
    day_tokens = []

    current_events = 0
    current_days = 0
    current_tokens = 0

    for line, length in zip(lines, line_token_lengths):
        current_tokens += length

        line_stripped = line.strip()
        if not line_stripped:
            continue

        if not line_stripped.startswith("#"):
            current_events += 1
            event_tokens.append(current_tokens)
        else:
            date_match = DATE_PATTERN.search(line_stripped)
            if date_match:
                current_days += int(date_match.group("days"))
                day_x.append(current_days)
                day_tokens.append(current_tokens)

    return {"event_tokens": event_tokens, "day_x": day_x, "day_tokens": day_tokens}


def plot_trajectory_metrics(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer):
    print("Calculating trajectory metrics...")

    nl_traj_ds = nl_raw.map(
        extract_trajectory, fn_kwargs={"tokenizer": nl_tokenizer, "split_pattern": "\n\n"}, num_proc=NUM_WORKERS
    )
    struct_traj_ds = struct_raw.map(
        extract_trajectory, fn_kwargs={"tokenizer": struct_tokenizer, "split_pattern": "//\n\n//"}, num_proc=NUM_WORKERS
    )

    print("Plotting Tokens vs Events...")

    # Use zero-copy numpy array concatenation for memory efficiency and speed
    nl_event_tokens = np.concatenate(nl_traj_ds["event_tokens"])
    nl_events = np.concatenate([np.arange(1, len(x) + 1) for x in nl_traj_ds["event_tokens"]])

    struct_event_tokens = np.concatenate(struct_traj_ds["event_tokens"])
    struct_events = np.concatenate([np.arange(1, len(x) + 1) for x in struct_traj_ds["event_tokens"]])

    df_events = pd.DataFrame(
        {
            "Dataset": ["Natural Language"] * len(nl_events) + ["Structured"] * len(struct_events),
            "Events": np.concatenate([nl_events, struct_events]),
            "Tokens": np.concatenate([nl_event_tokens, struct_event_tokens]),
        }
    )

    events_traj_path = EVAL_PLOTS_DIR / "tokens_vs_events.pdf"
    fig, ax1 = plt.subplots(figsize=(LINEWIDTH, 4))
    ax2 = ax1.twinx()

    for dataset, color in [("Natural Language", NL_COLOR), ("Structured", STRUCT_COLOR)]:
        df_sub = df_events[df_events["Dataset"] == dataset]
        if not df_sub.empty:
            grouped = df_sub.groupby("Events")["Tokens"].agg(["mean", "std", "count"]).reset_index()
            ax1.plot(grouped["mean"], grouped["Events"], color=color, label=dataset)

            # Fill standard deviation, ensuring we don't go below 0
            std_vals = grouped["std"].fillna(0)
            lower = (grouped["mean"] - std_vals).clip(lower=0)
            upper = grouped["mean"] + std_vals
            ax1.fill_betweenx(grouped["Events"], lower, upper, color=color, alpha=0.2)

            # Plot count on secondary axis
            ax2.fill_between(grouped["mean"], 0, grouped["count"], color=color, alpha=0.1)
            ax2.plot(grouped["mean"], grouped["count"], color=color, linestyle="--", alpha=0.5)

    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axvline(x=CONTEXT_WINDOW, color="red", linestyle="--", label=f"{CONTEXT_WINDOW} Context Window")
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Events")
    ax2.set_ylabel("Number of Sequences", color="gray")

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(events_traj_path)
    plt.close()
    print(f"Saved Tokens vs Events plot to {events_traj_path}")

    print("Plotting Tokens vs Days...")
    nl_day_x = np.concatenate(nl_traj_ds["day_x"]) if len(nl_traj_ds) > 0 else np.array([])
    nl_day_tokens = np.concatenate(nl_traj_ds["day_tokens"]) if len(nl_traj_ds) > 0 else np.array([])

    struct_day_x = np.concatenate(struct_traj_ds["day_x"]) if len(struct_traj_ds) > 0 else np.array([])
    struct_day_tokens = np.concatenate(struct_traj_ds["day_tokens"]) if len(struct_traj_ds) > 0 else np.array([])

    df_days = pd.DataFrame(
        {
            "Dataset": ["Natural Language"] * len(nl_day_x) + ["Structured"] * len(struct_day_x),
            "Days": np.concatenate([nl_day_x, struct_day_x])
            if len(nl_day_x) > 0 or len(struct_day_x) > 0
            else np.array([]),
            "Tokens": np.concatenate([nl_day_tokens, struct_day_tokens])
            if len(nl_day_tokens) > 0 or len(struct_day_tokens) > 0
            else np.array([]),
        }
    )

    days_traj_path = EVAL_PLOTS_DIR / "tokens_vs_days.pdf"
    fig, ax1 = plt.subplots(figsize=(LINEWIDTH, 4))
    ax2 = ax1.twinx()

    for dataset, color in [("Natural Language", NL_COLOR), ("Structured", STRUCT_COLOR)]:
        df_sub = df_days[df_days["Dataset"] == dataset]
        if not df_sub.empty:
            grouped = df_sub.groupby("Days")["Tokens"].agg(["mean", "std", "count"]).reset_index()
            ax1.plot(grouped["mean"], grouped["Days"], color=color, label=dataset)

            std_vals = grouped["std"].fillna(0)
            lower = (grouped["mean"] - std_vals).clip(lower=0)
            upper = grouped["mean"] + std_vals
            ax1.fill_betweenx(grouped["Days"], lower, upper, color=color, alpha=0.2)

            # Plot count on secondary axis
            ax2.fill_between(grouped["mean"], 0, grouped["count"], color=color, alpha=0.1)
            ax2.plot(grouped["mean"], grouped["count"], color=color, linestyle="--", alpha=0.5)

    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axvline(x=CONTEXT_WINDOW, color="red", linestyle="--", label=f"{CONTEXT_WINDOW} Context Window")
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Days")
    ax2.set_ylabel("Number of Sequences", color="gray")

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(days_traj_path)
    plt.close()
    print(f"Saved Tokens vs Days plot to {days_traj_path}")


def main():
    print("Loading tokenizers...")
    nl_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    struct_tokenizer = AutoTokenizer.from_pretrained(str(STRUCT_PRE_TRAINING_DIR / "tokenizer"))

    print("Loading raw datasets...")
    nl_raw = load_dataset("parquet", data_files=str(NL_PRE_TRAINING_DIR / "raw_data/train/*.parquet"), split="train")
    struct_raw = load_dataset(
        "parquet", data_files=str(STRUCT_PRE_TRAINING_DIR / "raw_data/train/*.parquet"), split="train"
    )

    # plot_header_counts()
    # plot_cdf_sequence_lengths(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer)
    plot_events_and_timeline_boxplots(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer)
    # plot_trajectory_metrics(nl_raw, struct_raw, nl_tokenizer, struct_tokenizer)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
