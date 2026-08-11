import argparse
import logging
import math
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
        "figure.figsize": (5.5, 4.5),  # Slightly taller than standard 3.5 to accommodate two subplots
        "figure.dpi": 300,
    }
)


def format_task_label(label):
    """Formats the task string nicely for the legend."""
    if not isinstance(label, str):
        return str(label)

    def process_word(w):
        return "ED" if w.lower() == "ed" else (w.title() if w.islower() else w)

    return " ".join(process_word(w) for w in label.replace("_", " ").split())


def _bootstrap_worker(seeds, y_true, y_prob, bin_idx_arr, cohort_task_indices, num_bins):
    """Worker function for bootstrapping AUROC per bin."""
    n_cohort = len(cohort_task_indices)
    results = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        boot_subj_idx = rng.randint(0, n_cohort, size=n_cohort)

        indices_list = [cohort_task_indices[i] for i in boot_subj_idx if len(cohort_task_indices[i]) > 0]
        if not indices_list:
            results.append((seed, np.full(num_bins, np.nan)))
            continue

        indices = np.concatenate(indices_list)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        bin_idx_boot = bin_idx_arr[indices]

        # Calculate AUROC per bin
        auroc_per_bin = np.full(num_bins, np.nan)
        for b in range(num_bins):
            mask = bin_idx_boot == b
            if np.sum(mask) >= 30:  # minimum sample requirement
                y_t = y_true_boot[mask]
                if len(np.unique(y_t)) > 1:
                    auroc_per_bin[b] = roc_auc_score(y_t, y_prob_boot[mask])

        results.append((seed, auroc_per_bin))
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot AUROC vs input length")
    parser.add_argument("--result_dir", type=Path, help="Path to the directory containing the results.")
    args = parser.parse_args()

    result_dir = args.result_dir
    output_dir = Path(os.environ["EVAL_PLOTS_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        log.error(f"Directory {result_dir} does not exist.")
        return

    # Get all task directories
    tasks = [d.name for d in result_dir.iterdir() if d.is_dir()]

    task_dfs = {}
    all_lengths = []

    # 1. Load data for all tasks first to find the global max length
    for task in sorted(tasks):
        task_dir = result_dir / task

        # Find the latest timestamp directory
        subdirs = [d for d in task_dir.iterdir() if d.is_dir()]
        if not subdirs:
            continue

        latest_timestamp_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        file_path = latest_timestamp_dir / "results.parquet"

        if not file_path.exists():
            log.warning(f"No results.parquet found for {task}")
            continue

        log.info(f"Loading data for {task} from {file_path}...")
        df = pd.read_parquet(file_path)

        df = df.dropna(subset=["boolean_value", "probability", "len_input_ids"])
        if len(df) == 0:
            continue

        task_dfs[task] = df
        all_lengths.extend(df["len_input_ids"].tolist())

    if not all_lengths:
        log.error("No valid data found in any tasks.")
        return

    # Calculate max length from data and define dynamic bins
    global_max_len = int(np.max(all_lengths))
    log.info(f"Global max length: {global_max_len}")
    bin_size = 512

    # Round up max length to the nearest bin_size boundary
    max_len_rounded = ((global_max_len // bin_size) + 1) * bin_size
    bins = list(range(0, max_len_rounded + bin_size, bin_size))
    num_bins = len(bins) - 1

    # Create 2 subplots: Top for AUROC (larger), Bottom for counts (smaller)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Use standard prop cycle colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    min_auroc = 1.0
    max_auroc = 0.0

    # Bootstrapping setup
    n_bootstraps = 1000
    bootstrap_seeds = np.random.RandomState(42).randint(0, 2**31 - 1, size=n_bootstraps)

    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, n_bootstraps)

    chunk_size = math.ceil(n_bootstraps / num_cores) if num_cores > 0 else 1
    chunks = [bootstrap_seeds[i : i + chunk_size] for i in range(0, n_bootstraps, chunk_size)]

    # 2. Process and plot AUROC lines for each task
    for idx, (task, df) in enumerate(task_dfs.items()):
        color = colors[idx % len(colors)]

        # Bin the lengths into fixed-size bins up to global max
        df["len_bin"] = pd.cut(df["len_input_ids"], bins=bins, right=False)

        results = []
        for b, (name, group) in enumerate(df.groupby("len_bin", observed=False)):
            count = len(group)
            # Require at least 100 samples to compute a stable AUROC
            if count >= 100 and len(group["boolean_value"].unique()) > 1:
                auroc = roc_auc_score(group["boolean_value"], group["probability"])
            else:
                auroc = np.nan

            # Use the bin center for plotting so all points align exactly
            mid_point = name.mid if not pd.isna(name) else np.nan

            results.append({"bin_idx": b, "mid_point": mid_point, "auroc": auroc, "count": count})

        res_df = pd.DataFrame(results).dropna(subset=["auroc"]).sort_values("mid_point")

        if len(res_df) > 0:
            log.info(f"Bootstrapping {n_bootstraps} samples for {task}...")

            # Subject-level Bootstrapping
            y_true = df["boolean_value"].values.astype(int)
            y_prob = df["probability"].values.astype(float)
            bin_idx_arr = df["len_bin"].cat.codes.values
            subject_ids = df["subject_id"].values

            unique_subjects = np.unique(subject_ids)
            subj_to_indices = {subj: [] for subj in unique_subjects}
            for i, subj in enumerate(subject_ids):
                subj_to_indices[subj].append(i)
            cohort_task_indices = [np.array(subj_to_indices[subj], dtype=int) for subj in unique_subjects]

            args_list = [(chunk, y_true, y_prob, bin_idx_arr, cohort_task_indices, num_bins) for chunk in chunks]

            with multiprocessing.Pool(processes=num_cores) as pool:
                boot_results = pool.starmap(_bootstrap_worker, args_list)

            all_boot_aurocs = np.full((n_bootstraps, num_bins), np.nan)
            for chunk_res in boot_results:
                for seed, aurocs in chunk_res:
                    seed_idx = np.where(bootstrap_seeds == seed)[0][0]
                    all_boot_aurocs[seed_idx, :] = aurocs

            # Get 95% CIs
            lower_ci = np.nanpercentile(all_boot_aurocs, 2.5, axis=0)
            upper_ci = np.nanpercentile(all_boot_aurocs, 97.5, axis=0)

            valid_bin_indices = res_df["bin_idx"].values
            res_df["lower_ci"] = lower_ci[valid_bin_indices]
            res_df["upper_ci"] = upper_ci[valid_bin_indices]

            formatted_task = format_task_label(task)
            ax1.plot(
                res_df["mid_point"],
                res_df["auroc"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=formatted_task,
                color=color,
            )

            # Plot the shaded confidence interval
            ax1.fill_between(res_df["mid_point"], res_df["lower_ci"], res_df["upper_ci"], alpha=0.2, color=color)

            min_auroc = min(min_auroc, np.nanmin(res_df["lower_ci"]))
            max_auroc = max(max_auroc, np.nanmax(res_df["upper_ci"]))

    # 3. Background histogram (counts) on the bottom subplot
    ax2.hist(all_lengths, bins=bins, alpha=0.5, color="gray", label="Total Count")
    ax2.set_ylabel("Count (Log)")
    ax2.set_yscale("log")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Customize top plot (AUROC)
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(max(0, min_auroc - 0.05), min(1.0, max_auroc + 0.05))
    ax1.set_xlim(0, max_len_rounded)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Place legend above the plot, spanning horizontally.
    # ncol=2 avoids overlap in a 5.5 inch wide figure
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    # Common X-axis label
    ax2.set_xlabel("Input Length (Bin Center)")

    fig.tight_layout()

    # Save the plot
    output_path = output_dir / "auroc_vs_len.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    log.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
