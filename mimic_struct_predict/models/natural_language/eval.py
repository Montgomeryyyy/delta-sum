import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from omegaconf import DictConfig
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)


# 1. Define Expected Calibration Error (ECE)
def expected_calibration_error(
    y_true: npt.NDArray[np.number],
    y_prob: npt.NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """Calculates ECE using equal-width binning from 0.0 to 1.0."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)  # Ensure 1.0 stays in last bin

    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece


# 2. Plotting Functions
def generate_plots(
    y_true: npt.NDArray[np.number],
    y_prob: npt.NDArray[np.floating],
    eval_dir: Path,
) -> None:
    """Generates and saves AUC, AUPRC, Calibration, and Histogram plots."""

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(eval_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_val = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AUPRC = {auprc_val:.3f}", color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(eval_dir / "pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Calibration Curve ---
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model", color="purple")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(eval_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Histogram of Predicted Probabilities ---
    fixed_bins = np.linspace(0.0, 1.0, 11)
    y_prob_1 = y_prob[y_true == 1]
    y_prob_0 = y_prob[y_true == 0]

    plt.figure(figsize=(8, 6))
    plt.hist(
        [y_prob_0, y_prob_1],
        bins=fixed_bins,
        stacked=True,
        alpha=0.7,
        color=["red", "green"],
        label=["Actual: No (0)", "Actual: Yes (1)"],
        edgecolor="white",  # Adds a tiny border between bars for clarity
        linewidth=0.5,
    )

    plt.xlim(0.0, 1.0)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Stacked Histogram of Predicted Probabilities")
    plt.legend(loc="upper center")
    plt.grid(alpha=0.3)
    plt.savefig(eval_dir / "prob_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="eval",
)
def main(cfg: DictConfig) -> None:
    for task_dir in (Path(cfg.data_dir) / cfg.split).iterdir():
        if not task_dir.is_dir():
            continue

        # 3. Load Data
        df = pl.read_parquet(task_dir / "*.parquet")
        if "predicted_boolean_probability" in df.columns:
            df = df.rename(
                {
                    "predicted_boolean_probability": "prob",
                }
            )

        eval_dir = task_dir / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving outputs to: {eval_dir}")

        if "completion" in df.columns:
            df = df.with_columns(predicted_boolean_value=pl.col("completion").str.ends_with("Yes"))

            output_blocks = []

            for row in df[:20].iter_rows(named=True):
                block = (
                    f"PROMPT:\n{row['prompt']}\n\n"
                    f"{'-' * 80}\n\n"
                    f"COMPLETION:\n{row['completion']}\n\n"
                    f"Format Adherence: {row['format_adherence']} | Prob: {row['prob']}\n\n"
                    f"{'=' * 80}"
                )
                output_blocks.append(block)
            (eval_dir / "completions.txt").write_text("\n\n".join(output_blocks), encoding="utf-8")

        # Extract to NumPy for speed
        y_true = df["boolean_value"].to_numpy().astype(int)
        y_prob = df["prob"].to_numpy()
        y_pred = df["predicted_boolean_value"].to_numpy().astype(int)

        # 4. Generate Plots
        generate_plots(y_true, y_prob, eval_dir)

        # 5. Calculate Non-Bootstrapped Baseline Metrics
        n_positives = int(np.sum(y_true))
        prevalence_rate = float(np.mean(y_true))

        point_estimates = {
            "AUROC": roc_auc_score(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
            "Brier Score": brier_score_loss(y_true, y_prob),
            "ECE": expected_calibration_error(y_true, y_prob),
            "F1 Score": f1_score(y_true, y_pred),
            "Loss": log_loss(y_true, y_pred),
        }

        # 6. Bootstrapping (1000 iterations)
        n_iterations = 1000
        boot_results: dict[str, list[float]] = {
            "AUROC": [],
            "AUPRC": [],
            "Brier Score": [],
            "ECE": [],
            "F1 Score": [],
            "Loss": [],
        }

        for _ in range(n_iterations):
            # Sample with replacement
            indices = np.random.randint(0, len(y_true), size=len(y_true))
            y_t_boot = y_true[indices]
            y_p_boot = y_prob[indices]
            y_pred_boot = y_pred[indices]

            # Safeguard: ROC AUC requires both classes present in the bootstrap sample
            if len(np.unique(y_t_boot)) < 2:
                continue

            boot_results["AUROC"].append(roc_auc_score(y_t_boot, y_p_boot))
            boot_results["AUPRC"].append(average_precision_score(y_t_boot, y_p_boot))
            boot_results["Brier Score"].append(brier_score_loss(y_t_boot, y_p_boot))
            boot_results["ECE"].append(expected_calibration_error(y_t_boot, y_p_boot))
            boot_results["F1 Score"].append(f1_score(y_t_boot, y_pred_boot, zero_division=0))
            boot_results["Loss"].append(log_loss(y_t_boot, y_p_boot))

        # 7. Assemble Final Summary DataFrame
        summary_rows = [
            {"Metric": "Number of Samples", "Estimate": len(df), "CI_95_Lower": None, "CI_95_Upper": None},
            {"Metric": "Number of Positives", "Estimate": float(n_positives), "CI_95_Lower": None, "CI_95_Upper": None},
            {"Metric": "Prevalence Rate", "Estimate": prevalence_rate, "CI_95_Lower": None, "CI_95_Upper": None},
        ]

        # Add bootstrapped metrics with 2.5th and 97.5th percentiles
        for metric, est in point_estimates.items():
            if boot_results[metric]:  # Ensure we have successful bootstraps
                lower, upper = np.percentile(boot_results[metric], [2.5, 97.5])
            else:
                lower, upper = None, None

            summary_rows.append(
                {
                    "Metric": metric,
                    "Estimate": est,
                    "CI_95_Lower": lower,
                    "CI_95_Upper": upper,
                }
            )

        results_df = pl.DataFrame(summary_rows)
        log.info("\nResults Summary:")
        log.info(results_df)

        # 8. Write Dataframe Output
        csv_out = eval_dir / "metrics_summary.csv"
        results_df.write_csv(csv_out)
        log.info(f"\nSaved metrics summary to: {csv_out}")


if __name__ == "__main__":
    main()
