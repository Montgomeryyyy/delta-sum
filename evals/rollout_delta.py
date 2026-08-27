import argparse
import collections
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import brier_score_loss, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Configure matplotlib for NeurIPS style camera-ready plots
sns.set_theme(style="ticks", context="paper", font_scale=1.0)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
    }
)

METHODS = {
    "probability_0": "Non-occurrence ($y=0$)",
    "probability_0_5": "Uninformative prior ($y=0.5$)",
    "probability_prevalence": "Training Prevalence",
    "probability_drop_trajectory": "Drop trajectories",
}

# Okabe-Ito Colorblind-safe palette (NeurIPS/ICML standard, high contrast, clean greyscale reproduction)
COLORS = ["#56B4E9", "#0072B2", "#009E73", "#D55E00"]

# Standardized 2-line labels ensuring identical vertical baselines across all x-axis ticks
TASK_MAP = {
    "ed_72hr_reattendance": "ED 72-Hour\nReattendance",
    "ed_to_hospital_admission": "ED to Hosp.\nAdmission",
    "hospital_length_of_stay_leq_7_days": "Hospital LOS\n$\\leq$ 7 Days",
    "hospital_post_discharge_30_day_mortality": "30-Day\nMortality",
    "in_hospital_mortality": "In-Hospital\nMortality",
    "Macro Average": "Macro\nAverage",
}


def format_task(label: str) -> str:
    if label in TASK_MAP:
        return TASK_MAP[label]
    return " ".join(
        "ED" if word.lower() == "ed" else (word.title() if word.islower() else word)
        for word in label.replace("_", " ").split()
    )


def load_predictions(base_dir: Path, task: str, split: str, is_xgb: bool = False) -> pl.DataFrame | None:
    """Loads prediction parquets for XGBoost or Rollout models."""
    task_dir = base_dir / task if is_xgb else base_dir / split / task
    if not task_dir.exists():
        if not is_xgb and (base_dir / task / split).exists():
            task_dir = base_dir / task / split
        else:
            return None
    subdirs = [dir_path for dir_path in task_dir.iterdir() if dir_path.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda dir_path: dir_path.name)
    if is_xgb:
        file_path = latest / "best_trial" / f"{split}_predictions.parquet"
        return pl.read_parquet(file_path) if file_path.exists() else None
    files = list(latest.glob("*.parquet"))
    return pl.read_parquet(files) if files else None


def align_dfs(df_xgb: pl.DataFrame, df_roll: pl.DataFrame) -> pl.DataFrame | None:
    """Aligns predictions between XGBoost and Rollout by joining on patient/time keys."""
    keys = ["subject_id", "prediction_time", "boolean_value"]
    xgb = df_xgb.select([*keys, pl.col("predicted_boolean_probability").alias("prob_xgb")])
    roll_cols = [col for col in METHODS if col in df_roll.columns]
    if not roll_cols:
        return None
    roll = df_roll.select(
        [
            *keys,
            *[pl.col(col).list.get(0) if isinstance(df_roll[col].dtype, pl.List) else pl.col(col) for col in roll_cols],
        ]
    )
    return xgb.join(roll, on=keys, how="inner").drop_nulls()


def _bootstrap_worker(chunk_seeds, y_true, prob_xgb, roll_dict, cohort_task_indices, metric_name):
    """Worker computing paired deltas for a chunk of random seeds across the global cohort."""
    n_cohort = len(cohort_task_indices)
    metric_fn = roc_auc_score if metric_name == "auroc" else brier_score_loss
    results = []

    for seed in chunk_seeds:
        rng = np.random.RandomState(seed)
        # 1. Resample global cohort patient indices with replacement (ensures identical patient sampling across tasks)
        boot_subj_idx = rng.randint(0, n_cohort, size=n_cohort)

        # 2. Gather row indices for resampled patients
        boot_indices = []
        for subj_idx in boot_subj_idx:
            boot_indices.extend(cohort_task_indices[subj_idx])

        indices = np.array(boot_indices, dtype=np.int32)
        y_boot = y_true[indices]

        # 3. Skip if resampled cohort does not contain both positive and negative labels
        if len(np.unique(y_boot)) < 2:
            results.append(None)
            continue

        # 4. Compute metric score for XGBoost baseline on the resampled cohort
        xgb_score = metric_fn(y_boot, prob_xgb[indices])

        # 5. Compute paired delta = Rollout - XGBoost for each error-handling probability method
        deltas = {method: metric_fn(y_boot, roll_dict[method][indices]) - xgb_score for method in roll_dict}
        results.append(deltas)

    return results


def run_paired_bootstrap(
    merged: pl.DataFrame, cohort_ids: list, seeds: np.ndarray, metric_name: str, num_workers: int = 1
):
    """Runs parallel patient-level paired bootstrap to compute the distribution of deltas (Rollout - XGBoost).

    Args:
        merged: DataFrame containing aligned ground truth labels and predicted probabilities.
        cohort_ids: List of unique patient IDs defining the global evaluation split cohort.
        seeds: Fixed array of random seeds for reproducible bootstrap sampling across tasks.
        metric_name: Target metric ("auroc" or "brier_score").
        num_workers: Number of parallel worker processes.

    Returns:
        Dictionary mapping each error-handling method to an array of bootstrap deltas of shape (n_bootstraps,).
    """
    y_true = merged["boolean_value"].to_numpy().astype(int)
    prob_xgb = merged["prob_xgb"].to_numpy().astype(float)
    roll_dict = {method: merged[method].to_numpy().astype(float) for method in METHODS if method in merged.columns}

    # Step 1: Pre-index row indices for each patient in the global cohort as lightweight integer lists.
    subj_map = collections.defaultdict(list)
    for row_idx, subj_id in enumerate(merged["subject_id"].to_numpy()):
        subj_map[subj_id].append(row_idx)

    cohort_task_indices = [subj_map.get(subj_id, []) for subj_id in cohort_ids]

    # Step 2: Split seeds into balanced chunks for parallel execution
    n_cores = min(num_workers, len(seeds))
    chunk_sz = math.ceil(len(seeds) / n_cores) if n_cores > 0 else 1
    chunks = [seeds[start_idx : start_idx + chunk_sz] for start_idx in range(0, len(seeds), chunk_sz)]

    # Step 3: Run parallel bootstrap across worker pool
    with multiprocessing.Pool(processes=n_cores) as pool:
        worker_results = pool.starmap(
            _bootstrap_worker,
            [(chunk, y_true, prob_xgb, roll_dict, cohort_task_indices, metric_name) for chunk in chunks],
        )

    # Step 4: Aggregate chunk results into delta arrays (shape: n_bootstraps)
    deltas = {method: np.full(len(seeds), np.nan) for method in roll_dict}
    bootstrap_idx = 0
    for chunk_res in worker_results:
        for res in chunk_res:
            if res is not None:
                for method in roll_dict:
                    deltas[method][bootstrap_idx] = res[method]
            bootstrap_idx += 1

    return deltas


def calc_ci(arr: np.ndarray) -> dict[str, float]:
    valid_values = arr[~np.isnan(arr)]
    if len(valid_values) == 0:
        return {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}
    return {
        "mean": float(np.mean(valid_values)),
        "lower_ci": float(np.percentile(valid_values, 2.5)),
        "upper_ci": float(np.percentile(valid_values, 97.5)),
    }


def plot_bars(
    task_stats: dict[str, dict[str, dict]],
    macro_stats: dict[str, dict],
    metric: str,
    out_dir: Path,
    modality_name: str,
):
    """Generates a single plot with all tasks on one x-axis and error methods in the legend."""
    if not task_stats:
        return

    tasks = list(task_stats.keys())
    all_x = tasks + (["Macro Average"] if macro_stats else [])
    x_positions = np.arange(len(all_x))

    fig, ax = plt.subplots(figsize=(max(5.5, len(all_x) * 1.15), 3.2))

    methods = list(METHODS.keys())
    width = 0.8 / len(methods)
    y_min, y_max = 0.0, 0.0

    for method_idx, method in enumerate(methods):
        means, low_err, up_err = [], [], []
        for task in tasks:
            stat = task_stats[task].get(method, {})
            mean_val = stat.get("mean", np.nan)
            lower_ci = stat.get("lower_ci", np.nan)
            upper_ci = stat.get("upper_ci", np.nan)
            means.append(mean_val)
            low_err.append(mean_val - lower_ci if not np.isnan(mean_val) else 0.0)
            up_err.append(upper_ci - mean_val if not np.isnan(mean_val) else 0.0)

        if macro_stats:
            stat = macro_stats.get(method, {})
            mean_val = stat.get("mean", np.nan)
            lower_ci = stat.get("lower_ci", np.nan)
            upper_ci = stat.get("upper_ci", np.nan)
            means.append(mean_val)
            low_err.append(mean_val - lower_ci if not np.isnan(mean_val) else 0.0)
            up_err.append(upper_ci - mean_val if not np.isnan(mean_val) else 0.0)

        for mean_val, lower_err, upper_err in zip(means, low_err, up_err):
            if not np.isnan(mean_val):
                y_min = min(y_min, mean_val - lower_err)
                y_max = max(y_max, mean_val + upper_err)

        ax.bar(
            x_positions + (method_idx - (len(methods) - 1) / 2) * width,
            means,
            width * 0.90,
            yerr=[low_err, up_err],
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#222222"},
            label=METHODS[method],
            color=COLORS[method_idx % len(COLORS)],
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
            alpha=0.92,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5, zorder=0)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, zorder=4)
    if macro_stats:
        ax.axvline(len(tasks) - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_task(task) for task in all_x], rotation=0, ha="center")
    ax.set_ylabel(f"$\\Delta$ {metric} (Rollout - XGBoost)", fontweight="bold")

    pad = max(0.02, (y_max - y_min) * 0.08) if y_max != y_min else 0.05
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4 if len(all_x) >= 5 else 2,
        frameon=False,
        fontsize=7.5,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    sns.despine(ax=ax, trim=False)
    plt.tight_layout()

    modality_slug = modality_name.lower().replace(" ", "_")
    slug = f"delta_{metric.lower().replace(' ', '_')}_{modality_slug}"
    out_pdf = out_dir / f"{slug}.pdf"
    out_png = out_dir / f"{slug}.png"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close()
    log.info(f"Saved delta plot to {out_pdf} and {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate metric deltas between XGBoost and Rollouts.")
    parser.add_argument(
        "--metric", type=str, default="auroc", choices=["auroc", "brier_score"], help="Evaluation metric"
    )
    parser.add_argument("--split", type=str, default="held_out", choices=["train", "tuning", "held_out"])
    parser.add_argument("--n_bootstraps", type=int, default=10_000)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", 1)),
    )
    parser.add_argument("--tasks", nargs="+", default=os.environ.get("ACES_TASKS", "").split())
    parser.add_argument("--xgboost_dir", type=Path, default=Path(os.environ.get("MEDS_TAB_OUTPUT_MODEL_DIR", "")))
    parser.add_argument(
        "--struct_dir", type=Path, default=Path(os.environ.get("STRUCTURED_INFERENCE_DIR", "")) / "pt" / "rollout"
    )
    parser.add_argument(
        "--nl_dir", type=Path, default=Path(os.environ.get("NATURAL_LANGUAGE_INFERENCE_DIR", "")) / "cpt" / "rollout"
    )
    parser.add_argument("--meds_cohort_dir", type=Path, default=Path(os.environ.get("MIMICIV_MEDS_COHORT_DIR", "")))
    parser.add_argument("--output_dir", type=Path, default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "rollout_delta")
    args = parser.parse_args()
    log.info(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_ids = []
    shards_file = args.meds_cohort_dir / "metadata" / ".shards.json"
    if shards_file.exists():
        for shard_name, shard_subject_ids in json.loads(shards_file.read_text()).items():
            if shard_name.startswith(args.split):
                cohort_ids.extend(shard_subject_ids)

    seeds = np.random.RandomState(42).randint(0, 2**31 - 1, size=args.n_bootstraps)
    sources = {"Structured Rollout": args.struct_dir, "Natural Language Rollout": args.nl_dir}
    metric_display = "AUROC" if args.metric == "auroc" else "Brier Score"

    summary_rows = []

    for mod_name, mod_dir in sources.items():
        log.info(f"--- Evaluating {mod_name} ({metric_display}) ---")
        task_stats = {}
        boot_accum = collections.defaultdict(list)

        for task in args.tasks:
            xgb_df = load_predictions(args.xgboost_dir, task, args.split, is_xgb=True)
            roll_df = load_predictions(mod_dir, task, args.split, is_xgb=False)
            if xgb_df is None or roll_df is None:
                log.warning(f"Skipping task '{task}' - missing XGBoost or Rollout predictions")
                continue
            log.info(f"Loaded {len(xgb_df)} XGBoost prediction rows for task '{task}'")
            log.info(f"Loaded {len(roll_df)} Rollout prediction rows for task '{task}'")

            merged = align_dfs(xgb_df, roll_df)
            if merged is None or merged.height == 0 or xgb_df.height != merged.height:
                log.warning(f"Skipping task '{task}' - no overlapping predictions")
                continue
            log.info(f"Merged {len(merged)} prediction rows for task '{task}'")

            log.info(f"Processing task '{task}' ({merged.height} rows)")
            deltas = run_paired_bootstrap(merged, cohort_ids, seeds, args.metric, num_workers=args.num_workers)

            task_stats[task] = {method: calc_ci(deltas[method]) for method in deltas}

            for method in deltas:
                boot_accum[method].append(deltas[method])
                summary_rows.append(
                    {
                        "Modality": mod_name,
                        "Task": task,
                        "Error Handling Method": METHODS[method],
                        f"Delta {metric_display} Mean": task_stats[task][method]["mean"],
                        f"Delta {metric_display} 95% CI Lower": task_stats[task][method]["lower_ci"],
                        f"Delta {metric_display} 95% CI Upper": task_stats[task][method]["upper_ci"],
                        "N Samples": merged.height,
                    }
                )

        # Macro averages
        macro_stats = {}
        for method in METHODS:
            if boot_accum[method]:
                macro_arr = np.nanmean(np.array(boot_accum[method]), axis=0)
                macro_stats[method] = calc_ci(macro_arr)
                summary_rows.append(
                    {
                        "Modality": mod_name,
                        "Task": "Macro Average",
                        "Error Handling Method": METHODS[method],
                        f"Delta {metric_display} Mean": macro_stats[method]["mean"],
                        f"Delta {metric_display} 95% CI Lower": macro_stats[method]["lower_ci"],
                        f"Delta {metric_display} 95% CI Upper": macro_stats[method]["upper_ci"],
                        "N Samples": np.nan,
                    }
                )

        # Render single-axis plot with all tasks for this modality
        if task_stats:
            plot_bars(task_stats, macro_stats, metric_display, args.output_dir, mod_name)

    if summary_rows:
        df = pl.DataFrame(summary_rows)
        metric_slug = args.metric.lower()
        csv_path = args.output_dir / f"rollout_deltas_{metric_slug}.csv"
        md_path = args.output_dir / f"rollout_deltas_{metric_slug}.md"
        df.write_csv(csv_path)
        md_path.write_text(df.to_pandas().to_markdown(index=False))
        log.info(f"Saved summary tables to {csv_path}")


if __name__ == "__main__":
    main()
