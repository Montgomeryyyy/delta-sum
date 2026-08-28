import argparse
import collections
import logging
import math
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import brier_score_loss, roc_auc_score

from evals.plot_style import (
    DEFAULT_FIG_HEIGHT_INCHES,
    ERRORBAR_KW,
    METHODS,
    METHODS_PALETTE,
    MODALITY_STYLES,
    NEURIPS_TEXTWIDTH_INCHES,
    SEPARATOR_LINE_KW,
    TASK_MAP,
    ZERO_LINE_KW,
    apply_neurips_style,
    get_modality_legend_handles,
    save_figure,
    sort_tasks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Apply NeurIPS style across the module
apply_neurips_style()


def load_predictions(base_dir: Path, task: str, split: str, is_xgb: bool = False) -> pl.DataFrame:
    """Loads prediction parquets for XGBoost or Rollout models."""
    task_dir = base_dir / task if is_xgb else base_dir / split / task
    latest = max(d for d in task_dir.iterdir() if d.is_dir())
    if is_xgb:
        return pl.read_parquet(latest / "best_trial" / f"{split}_predictions.parquet")
    return pl.read_parquet(list(latest.glob("*.parquet")))


def align_dfs(df_xgb: pl.DataFrame, df_roll: pl.DataFrame) -> pl.DataFrame:
    """Aligns predictions between XGBoost and Rollout by joining on patient/time keys."""
    keys = ["subject_id", "prediction_time", "boolean_value"]
    xgb = df_xgb.select([*keys, pl.col("predicted_boolean_probability").alias("prob_xgb")])
    roll_cols = [pl.col(col).list.get(0) if isinstance(df_roll[col].dtype, pl.List) else pl.col(col) for col in METHODS]
    roll = df_roll.select([*keys, *roll_cols, pl.col("format_adherence")])
    return xgb.join(roll, on=keys, how="inner").drop_nulls(subset=[*keys, "prob_xgb", *METHODS])


def compute_valid_trajectory_mask(df: pl.DataFrame) -> np.ndarray:
    """Computes a boolean mask where True indicates at least one trajectory adhered to format."""
    return df["format_adherence"].list.any().to_numpy()


def _bootstrap_worker(
    chunk_seeds: np.ndarray,
    y_true: np.ndarray,
    prob_xgb: np.ndarray,
    roll_dict: dict[str, np.ndarray],
    patient_row_indices: list[list[int]],
    metric_name: str,
    drop_valid_mask: np.ndarray,
):
    """Worker computing paired deltas for a chunk of random seeds across patient clusters."""
    n_patients = len(patient_row_indices)
    metric_fn = roc_auc_score if metric_name == "auroc" else brier_score_loss
    results = []

    for seed in chunk_seeds:
        rng = np.random.RandomState(seed)
        boot_subj_idx = rng.randint(0, n_patients, size=n_patients)

        boot_indices = []
        for subj_idx in boot_subj_idx:
            boot_indices.extend(patient_row_indices[subj_idx])

        indices = np.array(boot_indices, dtype=np.int32)
        y_boot = y_true[indices]

        if len(np.unique(y_boot)) < 2:
            results.append(None)
            continue

        xgb_score = metric_fn(y_boot, prob_xgb[indices])

        deltas = {}
        for method, prob_roll in roll_dict.items():
            if method == "probability_drop_trajectory":
                drop_indices = indices[drop_valid_mask[indices]]
                y_boot_drop = y_true[drop_indices]
                if len(np.unique(y_boot_drop)) < 2:
                    deltas[method] = np.nan
                else:
                    xgb_score_drop = metric_fn(y_boot_drop, prob_xgb[drop_indices])
                    deltas[method] = metric_fn(y_boot_drop, prob_roll[drop_indices]) - xgb_score_drop
            else:
                deltas[method] = metric_fn(y_boot, prob_roll[indices]) - xgb_score

        results.append(deltas)

    return results


def run_paired_bootstrap(
    merged: pl.DataFrame,
    metric_name: str,
    drop_valid_mask: np.ndarray,
    n_bootstraps: int = 10_000,
    num_workers: int = 1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Runs parallel patient-level paired bootstrap to compute the distribution of deltas (Rollout - XGBoost)."""
    y_true = merged["boolean_value"].to_numpy().astype(int)
    prob_xgb = merged["prob_xgb"].to_numpy().astype(float)
    roll_dict = {method: merged[method].to_numpy().astype(float) for method in METHODS}

    subj_map = collections.defaultdict(list)
    for row_idx, subj_id in enumerate(merged["subject_id"].to_numpy()):
        subj_map[subj_id].append(row_idx)
    patient_row_indices = list(subj_map.values())

    seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_bootstraps)
    n_cores = min(num_workers, len(seeds))
    chunk_sz = math.ceil(len(seeds) / n_cores) if n_cores > 0 else 1
    chunks = [seeds[start_idx : start_idx + chunk_sz] for start_idx in range(0, len(seeds), chunk_sz)]

    with multiprocessing.Pool(processes=n_cores) as pool:
        worker_results = pool.starmap(
            _bootstrap_worker,
            [
                (chunk, y_true, prob_xgb, roll_dict, patient_row_indices, metric_name, drop_valid_mask)
                for chunk in chunks
            ],
        )

    deltas = {method: np.full(n_bootstraps, np.nan) for method in roll_dict}
    bootstrap_idx = 0
    for chunk_res in worker_results:
        for res in chunk_res:
            if res is not None:
                for method in roll_dict:
                    deltas[method][bootstrap_idx] = res.get(method, np.nan)
            bootstrap_idx += 1

    return deltas


def calc_ci(arr: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(arr)),
        "lower_ci": float(np.nanpercentile(arr, 2.5)),
        "upper_ci": float(np.nanpercentile(arr, 97.5)),
    }


def plot_combined_bars(
    all_modality_stats: dict[str, dict[str, dict[str, dict]]],
    metric: str,
    tasks: list[str],
    out_dir: Path,
):
    """Generates a combined bar plot comparing Structured and Natural Language rollouts side-by-side per task."""
    tasks = sort_tasks(tasks)
    modalities = list(all_modality_stats.keys())
    x_positions = np.arange(len(tasks))
    methods = list(METHODS.keys())
    n_methods = len(methods)
    n_mods = len(modalities)

    bar_width = 0.078
    inter_method_gap = 0.032
    pair_span = n_mods * bar_width

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXTWIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES))
    y_min, y_max = 0.0, 0.0

    for method_idx, method in enumerate(methods):
        method_center_offset = (method_idx - (n_methods - 1) / 2) * (pair_span + inter_method_gap)
        color = METHODS_PALETTE[method]

        for mod_idx, mod_name in enumerate(modalities):
            style = MODALITY_STYLES[mod_name]
            mod_offset = (mod_idx - (n_mods - 1) / 2) * bar_width
            bar_x = x_positions + method_center_offset + mod_offset

            means = [all_modality_stats[mod_name][task][method]["mean"] for task in tasks]
            low_err = [m - all_modality_stats[mod_name][task][method]["lower_ci"] for m, task in zip(means, tasks)]
            up_err = [all_modality_stats[mod_name][task][method]["upper_ci"] - m for m, task in zip(means, tasks)]

            for m, l_err, u_err in zip(means, low_err, up_err):
                y_min = min(y_min, m - l_err)
                y_max = max(y_max, m + u_err)

            ax.bar(
                bar_x,
                means,
                bar_width * 0.92,
                yerr=[low_err, up_err],
                error_kw=ERRORBAR_KW,
                color=color,
                hatch=style["hatch"],
                zorder=3,
                alpha=style["alpha"],
                edgecolor="#222222",
                linewidth=0.5,
            )

    # Subtle separator lines between task clusters
    for sep_x in x_positions[:-1] + 0.5:
        ax.axvline(sep_x, **SEPARATOR_LINE_KW)

    ax.axhline(0, **ZERO_LINE_KW)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([TASK_MAP[task] for task in tasks], rotation=0, ha="center")
    ax.set_ylabel(f"$\\Delta$ {metric} (Rollout $-$ XGBoost)")

    pad = max(0.015, (y_max - y_min) * 0.12)
    ax.set_ylim(y_min - pad, y_max + pad)

    # Clean 2-row legend matching error_rates.py:
    # Row 1 = Methods (colored patches), Row 2 = Modalities (S solid, NL hatched)
    method_handles = [
        Patch(
            facecolor=METHODS_PALETTE[method],
            edgecolor="#222222",
            linewidth=0.5,
            label=METHODS[method],
        )
        for method in methods
    ]
    modality_handles = get_modality_legend_handles(facecolor="white", include_prefix=False)
    legend_handles = method_handles + modality_handles

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
    )

    sns.despine(ax=ax, trim=False)
    plt.tight_layout()

    metric_slug = metric.lower().replace(" ", "_")
    stem = f"delta_{metric_slug}_rollout"
    save_figure(out_dir, stem)
    plt.close()
    log.info(f"Saved combined delta plot to {out_dir / f'{stem}.png'}")


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
    parser.add_argument("--output_dir", type=Path, default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "rollout_delta")
    args = parser.parse_args()
    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"Structured Rollout": args.struct_dir, "Natural Language Rollout": args.nl_dir}
    metric_display = "AUROC" if args.metric == "auroc" else "Brier Score"

    summary_rows = []
    all_modality_stats = {}

    for mod_name, mod_dir in sources.items():
        log.info(f"--- Evaluating {mod_name} ({metric_display}) ---")
        task_stats = {}

        for task in args.tasks:
            xgb_df = load_predictions(args.xgboost_dir, task, args.split, is_xgb=True)
            roll_df = load_predictions(mod_dir, task, args.split, is_xgb=False)

            merged = align_dfs(xgb_df, roll_df)
            assert merged.height == xgb_df.height == roll_df.height, f"Row count mismatch for task '{task}'"

            drop_valid_mask = compute_valid_trajectory_mask(merged)
            n_dropped = int(np.sum(~drop_valid_mask))

            log.info(f"Processing task '{mod_name} {task}' ({merged.height} rows)")
            deltas = run_paired_bootstrap(
                merged,
                metric_name=args.metric,
                drop_valid_mask=drop_valid_mask,
                n_bootstraps=args.n_bootstraps,
                num_workers=args.num_workers,
            )

            task_stats[task] = {method: calc_ci(deltas[method]) for method in METHODS}

            for method in METHODS:
                n_samples = merged.height - n_dropped if method == "probability_drop_trajectory" else merged.height
                summary_rows.append(
                    {
                        "Modality": mod_name,
                        "Task": task,
                        "Error Handling Method": METHODS[method],
                        f"Delta {metric_display} Mean": task_stats[task][method]["mean"],
                        f"Delta {metric_display} 95% CI Lower": task_stats[task][method]["lower_ci"],
                        f"Delta {metric_display} 95% CI Upper": task_stats[task][method]["upper_ci"],
                        "N Samples": n_samples,
                    }
                )

        all_modality_stats[mod_name] = task_stats

    # Render unified combined plot comparing modalities across all tasks
    plot_combined_bars(all_modality_stats, metric_display, args.tasks, args.output_dir)

    df = pl.DataFrame(summary_rows)
    metric_slug = args.metric.lower()
    csv_path = args.output_dir / f"rollout_deltas_{metric_slug}.csv"
    md_path = args.output_dir / f"rollout_deltas_{metric_slug}.md"
    df.write_csv(csv_path)
    md_path.write_text(df.to_pandas().to_markdown(index=False))
    log.info(f"Saved summary tables to {csv_path}")


if __name__ == "__main__":
    main()
