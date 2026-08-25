import logging
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
log = logging.getLogger(__name__)

# Configure matplotlib for NeurIPS style plots
plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 300,
    }
)


def format_x_label(label):
    if not isinstance(label, str):
        return str(label)

    def process_word(w):
        return "ED" if w.lower() == "ed" else (w.title() if w.islower() else w)

    return " ".join(process_word(w) for w in label.replace("_", " ").split())


def load_task_results(base_dir, task, split):
    task_dir = base_dir / split / task
    if not task_dir.exists():
        log.info(f"Directory {task_dir} does not exist.")
        return None

    subdirs = [d for d in task_dir.iterdir() if d.is_dir()]
    if not subdirs:
        log.info(f"No run directories found in {task_dir}.")
        return None

    latest_dir = max(subdirs, key=lambda d: d.name)
    log.info(f"Loading results from {latest_dir}")
    results_files = list(latest_dir.glob("*.parquet"))
    if not results_files:
        log.info(f"No parquet files found in {latest_dir}. Skipping.")
        return None

    return pl.read_parquet(results_files)


def get_tasks(base_dir, split):
    split_dir = base_dir / split
    if not split_dir.exists():
        return []
    return [d.name for d in split_dir.iterdir() if d.is_dir()]


def process_data(df, task):
    # Explode the lists so each element becomes a separate row
    df = df.explode(["format_adherence", "finish_reason", "num_events"])

    # We want a column for task to keep track after combining
    df = df.with_columns(pl.lit(task).alias("task"))
    return df


def plot_error_rates(df, output_dir):
    # Error rate = 1 - format_adherence
    total_counts = df.group_by(["task", "model"]).agg(pl.len().alias("total"))

    errors_df = df.filter(~pl.col("format_adherence"))

    models = ["Structured", "Natural Language"]

    for model in models:
        log.info(f"--- {model} Finish Reasons for format_adherence == False ---")
        model_errors = errors_df.filter(pl.col("model") == model)
        if model_errors.height > 0:
            reason_counts = model_errors.group_by("finish_reason").agg(pl.len().alias("count"))
            for row in reason_counts.iter_rows(named=True):
                log.info(f"Reason: {row['finish_reason']}, Count: {row['count']}")

    error_types = errors_df.group_by(["task", "model", "finish_reason"]).agg(pl.len().alias("count"))

    # Join to get proportions
    error_types = error_types.join(total_counts, on=["task", "model"])
    error_types = error_types.with_columns((pl.col("count") / pl.col("total")).alias("proportion"))

    # Compute overall proportions
    overall_total = df.group_by("model").agg(pl.len().alias("total"))
    overall_errors = errors_df.group_by(["model", "finish_reason"]).agg(pl.len().alias("count"))
    overall_errors = overall_errors.join(overall_total, on="model")
    overall_errors = overall_errors.with_columns(pl.lit("Overall").alias("task"))
    overall_errors = overall_errors.with_columns((pl.col("count") / pl.col("total")).alias("proportion"))

    # Combine task-specific and overall
    plot_df = pl.concat(
        [
            error_types.select(["task", "model", "finish_reason", "proportion"]),
            overall_errors.select(["task", "model", "finish_reason", "proportion"]),
        ]
    )

    tasks = sorted([t for t in plot_df["task"].unique() if t != "Overall"]) + ["Overall"]
    reasons = sorted(plot_df["finish_reason"].unique().to_list())

    # Okabe-Ito colorblind friendly palette
    colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#000000"]

    x = np.arange(len(tasks))
    width = 0.35

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    for i, model in enumerate(models):
        offset = -width / 2 if i == 0 else width / 2
        bottoms = np.zeros(len(tasks))

        for j, reason in enumerate(reasons):
            proportions = []
            reason_data = plot_df.filter((pl.col("finish_reason") == reason) & (pl.col("model") == model))
            for t in tasks:
                match = reason_data.filter(pl.col("task") == t)
                proportions.append(match["proportion"][0] if match.height > 0 else 0.0)

            label = reason if i == 0 else ""
            ax.bar(
                x + offset,
                proportions,
                width * 0.95,
                bottom=bottoms,
                label=label,
                color=colors[j % len(colors)],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            bottoms += np.array(proportions)

    if "Overall" in tasks:
        ax.axvline(x=tasks.index("Overall") - 0.5, color="gray", linestyle="--", alpha=0.5, zorder=1)

    ax.set_xlabel("Task")
    ax.set_ylabel("Format Error Rate")

    wrapped_tasks = [textwrap.fill(format_x_label(t), width=15) for t in tasks]

    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_tasks, rotation=0, ha="center")
    ax.tick_params(axis="x", which="major", pad=15)

    minor_xticks = [pos + offset for pos in x for offset in (-width / 2, width / 2)]
    minor_xlabels = ["S", "NL"] * len(x)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels(minor_xlabels, minor=True, fontsize=8)
    ax.tick_params(axis="x", which="minor", length=0, pad=2)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Error Type", bbox_to_anchor=(1.05, 0.5), loc="center left")

    plt.tight_layout()
    output_file = output_dir / "combined_error_rates.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    log.info(f"Saved error plot to {output_dir}")


def plot_num_events_vs_adherence(df, output_dir, bin_size=32, last_bin_start=224):
    df = df.filter(pl.col("num_events").is_not_null())
    if df.height == 0:
        log.warning("No valid num_events data")
        return

    df = df.with_columns((1.0 - pl.col("format_adherence").cast(pl.Float32)).alias("error_rate"))

    # Bin by bin_size, capping the last bin at last_bin_start (all values >= last_bin_start fall into this bin)
    df = df.with_columns(
        pl.when(pl.col("num_events") >= last_bin_start)
        .then(last_bin_start)
        .otherwise(pl.col("num_events") // bin_size * bin_size)
        .alias("bin_start")
    )

    grouped = (
        df.group_by(["bin_start", "model"]).agg([pl.mean("error_rate"), pl.len().alias("count")]).sort("bin_start")
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    bins = list(range(0, last_bin_start + bin_size, bin_size))
    bin_labels = [f"{b}-{b+bin_size}" for b in bins[:-1]] + [f"{last_bin_start}+"]

    models = sorted(grouped["model"].unique())
    n_models = len(models)
    width = 0.8 / max(1, n_models)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, model in enumerate(models):
        model_data = grouped.filter(pl.col("model") == model)

        data_dict = dict(zip(model_data["bin_start"], model_data["error_rate"]))
        plot_means = [data_dict.get(b, np.nan) for b in bins]
        ax1.plot(
            range(len(bins)),
            plot_means,
            marker="o",
            label=model,
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
        )

        data_dict_counts = dict(zip(model_data["bin_start"], model_data["count"]))
        plot_counts = [data_dict_counts.get(b, 0) for b in bins]
        offset = (i - (n_models - 1) / 2) * width
        x_pos = [x + offset for x in range(len(bins))]
        ax2.bar(x_pos, plot_counts, width=width, label=model, color=colors[i % len(colors)], alpha=0.8)

    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel("Number of Predicted Events (Binned)")

    ax1.set_ylabel("Error Rate")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7, axis="y")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    outpath = output_dir / "combined_error_rate_vs_num_events.pdf"
    plt.savefig(outpath)
    plt.close()
    log.info(f"Saved combined trend plot to {outpath}")


def load_model_data(name, base_dir, split):
    log.info(f"--- Loading {name} data ---")

    tasks = get_tasks(base_dir, split)
    if not tasks:
        log.warning(f"No tasks found for {name} in {base_dir / split}")
        return None

    log.info(f"Tasks: {tasks}")

    all_dfs = []
    for task in tasks:
        df = load_task_results(base_dir, task, split)
        if df is not None:
            df = df.select(["format_adherence", "finish_reason", "num_events"])
            processed_df = process_data(df, task)
            all_dfs.append(processed_df)
        else:
            log.warning(f"No data found for {name} on task {task}")

    if not all_dfs:
        log.warning(f"No valid data found for {name}")
        return None

    combined_df = pl.concat(all_dfs)

    return combined_df.with_columns(pl.lit(name).alias("model"))


def main():
    split = "held_out"

    nl_dir = Path(os.environ["NATURAL_LANGUAGE_INFERENCE_DIR"]) / "cpt" / "rollout"
    struct_dir = Path(os.environ["STRUCTURED_INFERENCE_DIR"]) / "pt" / "rollout"

    eval_plots_dir = Path(os.environ["EVAL_PLOTS_DIR"]) / "rollout"
    eval_plots_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading Natural Language rollout from {nl_dir}")
    nl_df = load_model_data("Natural Language", nl_dir, split)

    log.info(f"Loading Structured rollout from {struct_dir}")
    struct_df = load_model_data("Structured", struct_dir, split)

    all_dfs = []
    if nl_df is not None:
        all_dfs.append(nl_df)
    if struct_df is not None:
        all_dfs.append(struct_df)

    if all_dfs:
        combined_all = pl.concat(all_dfs)
        plot_error_rates(combined_all, eval_plots_dir)
        # plot_num_events_vs_adherence(combined_all, eval_plots_dir)


if __name__ == "__main__":
    main()
