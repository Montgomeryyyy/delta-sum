import collections
import json
import logging
import os
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

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

RNG = np.random.RandomState(42)


def load_task_results(model_name, task, split, base_dir):
    # MEDS_TAB has named parquet files instead of directory structure
    task_dir = base_dir / task if model_name == "MEDS-TAB" else base_dir / split / task
    if not task_dir.exists():
        log.info(f"Directory {task_dir} does not exist. Skipping model '{model_name}'.")
        return None

    # Take most recent directory
    subdirs = [d for d in task_dir.iterdir() if d.is_dir()]
    if not subdirs:
        log.info(f"No run directories found in {task_dir}. Skipping.")
        return None
    latest_dir = max(subdirs, key=lambda d: d.name)

    results_file = (
        latest_dir / "best_trial" / f"{split}_predictions.parquet"
        if model_name == "MEDS-TAB"
        else latest_dir / "results.parquet"
    )
    if not results_file.exists():
        log.info(f"Results file {results_file} not found. Skipping.")
        return None

    # Extract boolean_value and probability
    return pl.read_parquet(results_file)


def format_x_label(label):
    if not isinstance(label, str):
        return str(label)

    def process_word(w):
        return "ED" if w.lower() == "ed" else (w.title() if w.islower() else w)

    return " ".join(process_word(w) for w in label.replace("_", " ").split())


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE) using sklearn's calibration_curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_total, _ = np.histogram(y_prob, bins=bin_edges)
    nonzero = bin_total > 0
    ece = np.sum((bin_total[nonzero] / len(y_true)) * np.abs(prob_true - prob_pred))
    return ece


def calculate_metrics(y_true, y_prob):
    """Calculates AUROC, AUPRC, Brier Score, and ECE."""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    return {"AUROC": auroc, "AUPRC": auprc, "Brier Score": brier, "ECE": ece}


def bootstrap_metrics(y_true, y_prob, task_subject_ids, cohort_subject_ids, bootstrap_seeds):
    """Generates bootstrapped confidence intervals for all metrics and curves."""
    n_bootstraps = len(bootstrap_seeds)
    subj_to_indices = collections.defaultdict(list)
    for i, subj in enumerate(task_subject_ids):
        subj_to_indices[subj].append(i)

    bootstrapped_results = {
        "AUROC": np.full(n_bootstraps, np.nan),
        "AUPRC": np.full(n_bootstraps, np.nan),
        "Brier Score": np.full(n_bootstraps, np.nan),
        "ECE": np.full(n_bootstraps, np.nan),
    }

    mean_fpr = np.linspace(0, 1, 100)
    tprs = np.full((n_bootstraps, 100), np.nan)

    mean_recall = np.linspace(0, 1, 100)
    precisions = np.full((n_bootstraps, 100), np.nan)

    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    prob_trues = np.full((n_bootstraps, 10), np.nan)

    for b_i, seed in enumerate(bootstrap_seeds):
        rng = np.random.RandomState(seed)
        boot_subject_ids = rng.choice(cohort_subject_ids, size=len(cohort_subject_ids), replace=True)

        indices_list_for_boot = [subj_to_indices[subj] for subj in boot_subject_ids if subj in subj_to_indices]
        if not indices_list_for_boot:
            log.warning(f"Bootstrap {b_i} has no valid subjects, skipping.")
            continue

        indices = np.concatenate(indices_list_for_boot)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        # Ensure we have both positive and negative samples
        if len(np.unique(y_true_boot)) < 2:
            log.warning(f"Bootstrap {b_i} has only one class, skipping.")
            continue

        metrics = calculate_metrics(y_true_boot, y_prob_boot)
        for k, v in metrics.items():
            bootstrapped_results[k][b_i] = v

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_boot, y_prob_boot)
        tprs[b_i] = np.interp(mean_fpr, fpr, tpr)

        # PR curve
        precision, recall, _ = precision_recall_curve(y_true_boot, y_prob_boot)
        precisions[b_i] = np.interp(mean_recall, recall[::-1], precision[::-1])

        # Calibration curve
        prob_true = np.zeros(10)
        for i in range(10):
            mask = (y_prob_boot >= bins[i]) & (y_prob_boot < bins[i + 1])
            if i == 9:
                mask = (y_prob_boot >= bins[i]) & (y_prob_boot <= bins[i + 1])
            if np.sum(mask) > 0:
                prob_true[i] = np.mean(y_true_boot[mask])
            else:
                prob_true[i] = np.nan
        prob_trues[b_i] = prob_true

    summary = {}
    for k, v in bootstrapped_results.items():
        valid_v = v[~np.isnan(v)]
        if len(valid_v) > 0:
            summary[k] = {
                "mean": np.mean(valid_v),
                "lower_ci": np.percentile(valid_v, 2.5),
                "upper_ci": np.percentile(valid_v, 97.5),
            }
        else:
            summary[k] = {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}

    def get_curve_stats(x, y_data):
        if len(y_data) == 0:
            nan_array = np.nan * np.ones_like(x)
            return {"x": x, "y_mean": nan_array, "y_lower": nan_array, "y_upper": nan_array}
        return {
            "x": x,
            "y_mean": np.nanmean(y_data, axis=0),
            "y_lower": np.nanpercentile(y_data, 2.5, axis=0),
            "y_upper": np.nanpercentile(y_data, 97.5, axis=0),
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        curve_data = {
            "ROC": get_curve_stats(mean_fpr, tprs),
            "PR": get_curve_stats(mean_recall, precisions),
            "Calibration": get_curve_stats(bin_centers, prob_trues),
        }

    return summary, curve_data, bootstrapped_results


def plot_curves(results_dict, task, output_dir):
    """Plots and saves ROC, PR, and Calibration curves with confidence intervals in NeurIPS style."""
    output_dir = Path(output_dir)
    task_out_dir = output_dir / task
    task_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    plt.figure()
    for model_name, data in results_dict.items():
        curve = data["curve_data"]["ROC"]
        metrics = data["metrics_summary"]["AUROC"]
        mean_fpr = curve["x"]
        mean_tpr = curve["y_mean"]
        lower_tpr = curve["y_lower"]
        upper_tpr = curve["y_upper"]

        # Ensure mean_tpr starts at 0 and ends at 1
        if len(mean_tpr) > 0 and not np.isnan(mean_tpr[0]):
            mean_tpr[0] = 0.0
            mean_tpr[-1] = 1.0

        plt.plot(
            mean_fpr,
            mean_tpr,
            label=f'{model_name} (AUC = {metrics["mean"]:.3f} [{metrics["lower_ci"]:.3f}-{metrics["upper_ci"]:.3f}])',
        )
        plt.fill_between(mean_fpr, lower_tpr, upper_tpr, alpha=0.2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.tight_layout()
    plt.savefig(task_out_dir / "roc_curve.pdf")
    plt.close()

    # 2. PR Curve
    plt.figure()
    for model_name, data in results_dict.items():
        curve = data["curve_data"]["PR"]
        metrics = data["metrics_summary"]["AUPRC"]
        mean_recall = curve["x"]
        mean_precision = curve["y_mean"]
        lower_precision = curve["y_lower"]
        upper_precision = curve["y_upper"]

        plt.plot(
            mean_recall,
            mean_precision,
            label=f'{model_name} (AP = {metrics["mean"]:.3f} [{metrics["lower_ci"]:.3f}-{metrics["upper_ci"]:.3f}])',
        )
        plt.fill_between(mean_recall, lower_precision, upper_precision, alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.tight_layout()
    plt.savefig(task_out_dir / "pr_curve.pdf")
    plt.close()

    # 3. Calibration Curve
    num_models = len(results_dict)
    num_hist_rows = (num_models + 1) // 2

    fig = plt.figure(figsize=(10, 6 + 2.5 * num_hist_rows))
    gs = fig.add_gridspec(num_hist_rows + 2, 2)

    ax_main = fig.add_subplot(gs[0:2, :])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (model_name, data) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        curve = data["curve_data"]["Calibration"]
        metrics = data["metrics_summary"]["Brier Score"]
        bin_centers = curve["x"]
        mean_prob_true = curve["y_mean"]
        lower_prob_true = curve["y_lower"]
        upper_prob_true = curve["y_upper"]

        ax_main.plot(
            bin_centers,
            mean_prob_true,
            marker="s",
            color=color,
            label=f'{model_name} (Brier = {metrics["mean"]:.3f} [{metrics["lower_ci"]:.3f}-{metrics["upper_ci"]:.3f}])',
        )
        ax_main.fill_between(bin_centers, lower_prob_true, upper_prob_true, alpha=0.2, color=color)

    ax_main.plot([0, 1], [0, 1], "k:", alpha=0.8, label="Perfectly calibrated")
    ax_main.set_xlabel("Mean predicted probability")
    ax_main.set_ylabel("Fraction of positives")
    ax_main.legend(loc="lower right")
    ax_main.set_xlim(left=0, right=1)
    ax_main.set_ylim(bottom=0, top=1)

    # Histograms
    for i, (model_name, data) in enumerate(results_dict.items()):
        row = (i // 2) + 2
        col = i % 2
        ax_hist = fig.add_subplot(gs[row, col])

        y_prob = data["y_prob"]
        color = colors[i % len(colors)]
        ax_hist.hist(y_prob, bins=20, range=(0, 1), color=color, log=True)
        ax_hist.set_title(model_name)
        ax_hist.set_xlabel("Mean predicted probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_xlim(left=0, right=1)

    fig.tight_layout()
    plt.savefig(task_out_dir / "calibration_curve.pdf")
    plt.close()


def plot_bar_charts(results_dict, task, output_dir):
    """Plots and saves bar charts for each metric comparing models with CIs."""
    output_dir = Path(output_dir)
    task_out_dir = output_dir / task
    task_out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["AUROC", "AUPRC", "Brier Score", "ECE"]

    for metric in metrics:
        plt.figure()

        models = list(results_dict.keys())
        means = []
        lower_errors = []
        upper_errors = []

        for m in models:
            stats = results_dict[m]["metrics_summary"][metric]
            means.append(stats["mean"])
            lower_errors.append(stats["mean"] - stats["lower_ci"])
            upper_errors.append(stats["upper_ci"] - stats["mean"])

        yerr = [lower_errors, upper_errors]

        x_pos = np.arange(len(models))

        plt.bar(x_pos, means, yerr=yerr, capsize=5, alpha=0.8)
        wrapped_models = [textwrap.fill(format_x_label(m), width=15) for m in models]
        plt.xticks(x_pos, wrapped_models, rotation=0, ha="center")
        plt.ylabel(metric)

        if metric in ["AUROC", "AUPRC"]:
            min_val = min([m - e for m, e in zip(means, lower_errors) if not np.isnan(m)], default=0)
            plt.ylim(max(0, min_val * 0.95), 1)
        elif metric in ["Brier Score", "ECE"]:
            plt.ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(task_out_dir / f'{metric.lower().replace(" ", "_")}_bar.pdf')
        plt.close()


def plot_overall_bar_charts(all_metrics, output_dir):
    """Plots and saves bar charts for each metric comparing models across all tasks."""
    output_dir = Path(output_dir)

    metrics = ["AUROC", "AUPRC", "Brier Score", "ECE"]

    # Extract unique tasks and models while preserving order
    tasks = list(dict.fromkeys(row["Task"] for row in all_metrics))
    models = list(dict.fromkeys(row["Model"] for row in all_metrics))

    for metric in metrics:
        plt.figure(figsize=(max(5.5, len(tasks) * 1.5), 3.5))

        x = np.arange(len(tasks))
        width = 0.8 / len(models)

        all_lower_cis = []

        for i, model in enumerate(models):
            means = []
            lower_errors = []
            upper_errors = []

            for task in tasks:
                # Find the row for this model and task
                row = next((r for r in all_metrics if r["Task"] == task and r["Model"] == model), None)
                if row:
                    mean_val = row[f"{metric} Mean"]
                    lower_ci = row[f"{metric} 95% CI Lower"]
                    upper_ci = row[f"{metric} 95% CI Upper"]
                    means.append(mean_val)
                    lower_errors.append(mean_val - lower_ci)
                    upper_errors.append(upper_ci - mean_val)
                else:
                    means.append(np.nan)
                    lower_errors.append(np.nan)
                    upper_errors.append(np.nan)

            all_lower_cis.extend([m - e for m, e in zip(means, lower_errors) if not np.isnan(m) and not np.isnan(e)])

            lower_errors = [0 if np.isnan(e) else e for e in lower_errors]
            upper_errors = [0 if np.isnan(e) else e for e in upper_errors]
            yerr = [lower_errors, upper_errors]

            plt.bar(
                x + i * width - width * (len(models) - 1) / 2,
                means,
                width,
                yerr=yerr,
                capsize=3,
                label=model,
                alpha=0.8,
            )

        plt.xlabel("Task")
        plt.ylabel(metric)

        wrapped_tasks = [textwrap.fill(format_x_label(t), width=15) for t in tasks]
        plt.xticks(x, wrapped_tasks, rotation=0, ha="center")
        plt.legend()

        if metric in ["AUROC", "AUPRC"]:
            min_val = min(all_lower_cis, default=0) if all_lower_cis else 0
            plt.ylim(max(0, min_val * 0.95), 1)
        elif metric in ["Brier Score", "ECE"]:
            plt.ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(output_dir / f'overall_{metric.lower().replace(" ", "_")}_bar.pdf')
        plt.close()


def main():
    # Variables populated from config.sh
    split = "tuning"
    n_bootstraps = 1000
    eval_plots_dir = Path(os.environ["EVAL_PLOTS_DIR"]) / f"metrics_{split}"
    meds_shards_metadata = Path(os.environ["MIMICIV_MEDS_COHORT_DIR"]) / "metadata" / ".shards.json"
    eval_plots_dir.mkdir(parents=True, exist_ok=True)
    aces_tasks = os.environ.get("ACES_TASKS").split()

    # The mapping from model name to the base directory for inference results
    base_model = os.environ["HF_MODEL_NAME"].split("/")[-1].title()
    model_to_directory_dict = {
        "MEDS-TAB": Path(os.environ["MEDS_TAB_OUTPUT_MODEL_DIR"]),
        "Structured (Random Init)": Path(os.environ["STRUCTURED_INFERENCE_DIR"]) / "pt" / "embedding_lr_predictions",
        base_model: Path(os.environ["NATURAL_LANGUAGE_INFERENCE_DIR"]) / "zero_shot" / "embedding_lr_predictions",
        f"CPT ({base_model})": Path(os.environ["NATURAL_LANGUAGE_INFERENCE_DIR"]) / "cpt" / "embedding_lr_predictions",
    }
    log.info("Model to directory dictionary: ")
    log.info(model_to_directory_dict)

    # Store bootstrapped results across tasks per model
    all_metrics = []

    cohort_subject_ids = []
    for shard_name, shard_subject_ids in json.loads(meds_shards_metadata.read_text()).items():
        if shard_name.startswith(split):
            cohort_subject_ids.extend(shard_subject_ids)

    # Generate random seeds for the bootstraps
    bootstrap_seeds = RNG.randint(0, 2**31 - 1, size=n_bootstraps)

    # Store data for macro average computation
    # model_name -> metric_name -> list of arrays of shape (n_bootstraps,)
    model_to_task_bootstraps = {
        model_name: {
            metric: np.full((len(aces_tasks), n_bootstraps), np.nan, dtype=float)
            for metric in ["AUROC", "AUPRC", "Brier Score", "ECE"]
        }
        for model_name in model_to_directory_dict.keys()
    }
    model_to_prevalence = {model_name: np.zeros(len(aces_tasks)) for model_name in model_to_directory_dict.keys()}

    for task_idx, task in enumerate(aces_tasks):
        log.info(f"--- Evaluating Task: {task} ---")

        results_dict = {}
        for model_name, base_dir in model_to_directory_dict.items():
            df = load_task_results(model_name, task, split, base_dir)
            if df is None:
                log.warning(f"No results found for '{model_name}' on task '{task}'. Skipping")
                continue

            y_true = df["boolean_value"].to_numpy().astype(int)
            prob_col = "predicted_boolean_probability" if model_name == "MEDS-TAB" else "probability"
            y_prob = df[prob_col].to_numpy().astype(float)
            subject_ids_for_task = df["subject_id"].to_numpy()

            log.info(f"Bootstrapping metrics and curves for '{model_name}' on task '{task}'...")
            metrics_summary, curve_data, bootstrapped_results = bootstrap_metrics(
                y_true, y_prob, subject_ids_for_task, cohort_subject_ids, bootstrap_seeds
            )

            for metric_name, bootstrap_array in bootstrapped_results.items():
                model_to_task_bootstraps[model_name][metric_name][task_idx, :] = bootstrap_array

            prevalence = np.mean(y_true)
            model_to_prevalence[model_name][task_idx] = prevalence

            row = {"Task": task, "Model": model_name, "Prevalence": prevalence}
            for metric_name, stats in metrics_summary.items():
                row[f"{metric_name} Mean"] = stats["mean"]
                row[f"{metric_name} 95% CI Lower"] = stats["lower_ci"]
                row[f"{metric_name} 95% CI Upper"] = stats["upper_ci"]
            all_metrics.append(row)

            results_dict[model_name] = {
                "metrics_summary": metrics_summary,
                "curve_data": curve_data,
                "y_prob": y_prob,
            }

        log.info(f"Generating plots for '{task}'...")
        plot_curves(results_dict, task, eval_plots_dir)
        plot_bar_charts(results_dict, task, eval_plots_dir)

    for model_name in model_to_directory_dict.keys():
        log.info(f"Calculating Macro Averages for {model_name} across tasks...")
        if not model_to_prevalence[model_name]:
            continue
        mean_prevalence = np.mean(model_to_prevalence[model_name])
        row = {"Task": "Macro Average", "Model": model_name, "Prevalence": mean_prevalence}
        for metric_name, task_bootstraps in model_to_task_bootstraps[model_name].items():
            # task_bootstraps is a numpy array of shape (n_tasks, n_bootstraps)
            macro_bootstraps = np.nanmean(task_bootstraps, axis=0)  # shape (n_bootstraps,)
            row[f"{metric_name} Mean"] = np.nanmean(macro_bootstraps)
            row[f"{metric_name} 95% CI Lower"] = np.nanpercentile(macro_bootstraps, 2.5)
            row[f"{metric_name} 95% CI Upper"] = np.nanpercentile(macro_bootstraps, 97.5)
        all_metrics.append(row)

    # Save overall metrics across all models and tasks
    log.info("Generating overall bar charts...")
    plot_overall_bar_charts(all_metrics, eval_plots_dir)

    log.info("Writing aggregated metrics to file...")
    all_metrics_df = pl.DataFrame(all_metrics)
    all_metrics_df.write_csv(eval_plots_dir / "all_metrics.csv")
    (eval_plots_dir / "all_metrics.md").write_text(all_metrics_df.to_pandas().to_markdown(index=False))
    log.info(f"Saved aggregated metrics to {eval_plots_dir}")


if __name__ == "__main__":
    main()
