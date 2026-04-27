# %%
import concurrent.futures
import textwrap
import zlib
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import bootstrap
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# %%
RANDOM_SEED = 42
N_BOOTSTRAP_SAMPLES = 100

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})

# %%
inference_dir = Path("../../outputs/inference")
split = "tuning"

data_to_method_to_results_dir = {
    "Tabular": {
        "MEDS-Tab": inference_dir / "tabular" / "20260427_161435" / split,
    },
    "Structured": {
        "Embedding + LR": inference_dir / "structured" / "embedding_and_lr" / "20260427_161537" / split,
        "Generative Rollout": inference_dir / "structured" / "generative_rollout" / "20260427_145042" / split,
    },
    "Natural Language": {
        "Qwen 0.6B Direct": inference_dir / "natural_language" / "direct" / "20260427_153012" / split,
        "Qwen 0.6B Thinking": inference_dir / "natural_language" / "thinking" / "20260427_153159" / split,
        "Qwen 0.6B SFT": inference_dir / "natural_language" / "direct_sft" / "20260427_154847" / split,
        # "natural_language_frontier_model":
    },
}
data_order = ["Tabular", "Structured", "Natural Language"]

# %%
dfs = []

for data, method_to_results_dir in data_to_method_to_results_dir.items():
    for method, results_dir in method_to_results_dir.items():
        for task_dir in results_dir.iterdir():
            task_results_df = pl.read_parquet(
                task_dir / "results.parquet",
                columns=["boolean_value", "probability", "sample_latency", "format_adherence"],
            )
            task_results_df = task_results_df.with_columns(
                pl.col("boolean_value").cast(pl.Float32),
                pl.col("probability").cast(pl.Float32),
                pl.col("sample_latency").cast(pl.Float32),
                pl.col("format_adherence").cast(pl.Float32),
            )

            task_results_df = task_results_df.with_columns(
                pl.lit(method).alias("method"),
                pl.lit(task_dir.name.replace("_", " ").title()).alias("task"),
                pl.lit(data).alias("data"),
            )
            dfs.append(task_results_df)

# Concatenate all DataFrames at once (much faster and avoids schema errors)
results_df = pl.concat(dfs, how="vertical")

# Print the final length
print(len(results_df))

print(results_df)

# %% [markdown]
# ## Latency

# %%
latency_results = []

for df_group in results_df.partition_by("method"):
    method_name = df_group["method"][0]
    latency_data = df_group["sample_latency"].to_numpy()

    # 2. Calculate the actual median (P50)
    p50 = np.median(latency_data)

    # 3. Use SciPy to calculate the Bootstrapped 95% CI for the median
    # Note: 'bootstrap' expects data as a sequence, so we pass it as a tuple: (latency_data,)
    res = bootstrap((latency_data,), np.median, confidence_level=0.95, method="BCa", random_state=RANDOM_SEED)

    # 4. Store the results
    latency_results.append(
        {
            "data": df_group["data"][0],
            "method": method_name,
            "p50_latency": p50,
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high,
        }
    )

# 5. Convert back to a Polars DataFrame and sort
bootstrap_summary_df = pl.DataFrame(latency_results)
mapping_dict = {name: i for i, name in enumerate(data_order)}
bootstrap_summary_df = (
    bootstrap_summary_df.with_columns(data_rank=pl.col("data").replace_strict(mapping_dict, return_dtype=pl.Int8))
    .sort("data_rank", "p50_latency")
    .drop("data_rank")
)


bootstrap_summary_df.to_pandas().to_markdown(OUTPUT_DIR / "latency.md", index=False)

print(bootstrap_summary_df)

# %%
# Convert to pandas just to make plotting the error bars easier
plot_df = bootstrap_summary_df.to_pandas()

data_to_color_and_marker = {
    "Tabular": ("#0072B2", "s"),
    "Structured": ("#E69F00", "o"),
    "Natural Language": ("#009E73", "D"),
}

# Calculate the distance from the median to the upper and lower bounds for the error bars
# Matplotlib expects error bars as [lower_errors, upper_errors] relative to the central point
yerr_lower = plot_df["p50_latency"] - plot_df["ci_lower"]
yerr_upper = plot_df["ci_upper"] - plot_df["p50_latency"]

plt.figure(figsize=(9, 6))

for label, (color, marker) in data_to_color_and_marker.items():
    mask = plot_df["data"] == label
    if mask.any():
        plt.errorbar(
            [textwrap.fill(method, 15) for method in plot_df.loc[mask, "method"]],
            plot_df.loc[mask, "p50_latency"],
            yerr=[yerr_lower[mask], yerr_upper[mask]],
            fmt=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1,
            color=color,  # Now receives a single string color per iteration
            ecolor="#444444",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="None",
            label=label,  # Added label for a legend
        )

plt.ylabel("Median Latency (s)")
plt.xlabel("Method")
plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.xticks(rotation=45, ha='right')
plt.yscale("log")
plt.legend(title="Data Type", loc="lower right")
plt.grid(axis="y", which="minor", linestyle=":", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latency.pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Format Adherence

# %%
latency_results = []

for df_group in results_df.partition_by("method"):
    method_name = df_group["method"][0]
    group_data = df_group["format_adherence"].to_numpy()

    # 2. Calculate the actual median (P50)
    p50 = np.median(group_data)

    # 3. Use SciPy to calculate the Bootstrapped 95% CI for the median
    # Note: 'bootstrap' expects data as a sequence, so we pass it as a tuple: (latency_data,)
    res = bootstrap((group_data,), np.median, confidence_level=0.95, method="BCa", random_state=RANDOM_SEED)

    # 4. Store the results
    latency_results.append(
        {
            "data": df_group.item(0, "data"),
            "method": method_name,
            "p50_latency": p50,
            "ci_lower": res.confidence_interval.low,
            "ci_upper": res.confidence_interval.high,
        }
    )

# 5. Convert back to a Polars DataFrame and sort
bootstrap_summary_df = pl.DataFrame(latency_results)
mapping_dict = {name: i for i, name in enumerate(data_order)}
bootstrap_summary_df = (
    bootstrap_summary_df.with_columns(data_rank=pl.col("data").replace_strict(mapping_dict, return_dtype=pl.Int8))
    .sort("data_rank", "p50_latency")
    .drop("data_rank")
)


bootstrap_summary_df.to_pandas().to_markdown(OUTPUT_DIR / "format_adherence.md", index=False)

print(bootstrap_summary_df)

# %%
# ==========================================
# 1. Core Metric Functions (Unchanged)
# ==========================================


def safe_metric_calc(y_true: np.ndarray, y_pred: np.ndarray, metric_func: Callable) -> float:
    try:
        # Check function name instead of object identity
        if metric_func.__name__ in ["roc_auc_score", "average_precision_score"]:
            # O(N) check: if min == max, there is only 1 unique class
            if y_true.min() == y_true.max():
                return np.nan
        return metric_func(y_true, y_pred)
    except Exception:
        return np.nan


def safe_macro_metric(
    y_true: np.ndarray, y_pred: np.ndarray, task_ids: np.ndarray, unique_tasks: np.ndarray, metric_func: Callable
) -> float:
    scores = []
    for t in unique_tasks:
        mask = task_ids == t
        if not mask.any():  # Skip if task didn't make it into this bootstrap sample
            continue
        scores.append(safe_metric_calc(y_true[mask], y_pred[mask], metric_func))
    if not scores:
        return np.nan
    return float(np.nanmean(scores))


# ==========================================
# 2. FAST Lightweight Bootstrap
# ==========================================


def fast_bootstrapped_ci(
    data_arrays: tuple, metric_func: Callable, n_bootstrap_samples: int, random_seed: int
) -> tuple:
    """A lightweight numpy bootstrap that avoids Scipy's massive overhead."""
    point_est = metric_func(*data_arrays)

    if np.isnan(point_est):
        return point_est, float("nan"), float("nan")

    n = len(data_arrays[0])
    rng = np.random.default_rng(random_seed)
    scores = np.empty(n_bootstrap_samples)

    for i in range(n_bootstrap_samples):
        # 1. Generate random indices once per iteration
        idx = rng.integers(0, n, size=n)
        # 2. Slice all input arrays using those indices
        resampled_data = [arr[idx] for arr in data_arrays]
        # 3. Calculate metric
        scores[i] = metric_func(*resampled_data)

    # Filter out NaNs (happens if a bootstrap sample only grabs 1 class)
    valid_scores = scores[~np.isnan(scores)]

    if len(valid_scores) < 2:
        return point_est, float("nan"), float("nan")

    ci_low = np.percentile(valid_scores, 2.5)
    ci_high = np.percentile(valid_scores, 97.5)

    return point_est, ci_low, ci_high


# ==========================================
# 3. Group Processing Function
# ==========================================


def calculate_all_metrics(
    group_df: pl.DataFrame, metric_name_to_function: dict[str, Callable], n_bootstrap_samples: int, random_seed: int
) -> pl.DataFrame:
    method = group_df["method"][0]
    data = group_df["data"][0]

    group_string = f"{method}_{data}".encode("utf-8")
    group_seed = random_seed + zlib.crc32(group_string)

    y_true = group_df["boolean_value"].to_numpy()
    y_pred = group_df["probability"].to_numpy()
    tasks = group_df["task"].to_numpy()

    results = []

    # --- A. Per-Task Metrics ---
    for task in np.unique(tasks):
        mask = tasks == task
        y_true_task, y_pred_task = y_true[mask], y_pred[mask]

        for metric_name, metric_func in metric_name_to_function.items():
            task_wrapper = partial(safe_metric_calc, metric_func=metric_func)
            metric, ci_low, ci_high = fast_bootstrapped_ci(
                (y_true_task, y_pred_task), task_wrapper, n_bootstrap_samples, group_seed
            )

            results.append(
                {
                    "data": data,
                    "method": method,
                    "task": str(task),
                    "metric": metric_name,
                    "value": metric,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "support": len(y_true_task),
                }
            )

    # --- B. Macro Metric ---
    for metric_name, metric_func in metric_name_to_function.items():
        macro_wrapper = partial(safe_macro_metric, metric_func=metric_func, unique_tasks=np.unique(tasks))
        metric_mac, ci_low_mac, ci_high_mac = fast_bootstrapped_ci(
            (y_true, y_pred, tasks), macro_wrapper, n_bootstrap_samples, group_seed
        )

        results.append(
            {
                "data": data,
                "method": method,
                "task": "MACRO",
                "metric": metric_name,
                "value": metric_mac,
                "ci_lower": ci_low_mac,
                "ci_upper": ci_high_mac,
                "support": len(y_true),
            }
        )

    return pl.DataFrame(results)


# ==========================================
# 4. Multiprocessed Execution
# ==========================================

if __name__ == "__main__":
    metric_name_to_function = {"AUROC": roc_auc_score, "AUPRC": average_precision_score, "Brier": brier_score_loss}

    # 1. Break the Polars dataframe into a list of smaller dataframes (one per group)
    grouped_dfs = (df for _, df in results_df.group_by("method", "data"))
    worker_func = partial(
        calculate_all_metrics,
        metric_name_to_function=metric_name_to_function,
        n_bootstrap_samples=N_BOOTSTRAP_SAMPLES,
        random_seed=RANDOM_SEED,
    )
    # 2. Process groups in parallel across all available CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processed_chunks = executor.map(worker_func, grouped_dfs)

    # 3. Recombine and format
    metrics_df = (
        pl.concat(processed_chunks)
        .with_columns(data_rank=pl.col("data").replace_strict(mapping_dict, return_dtype=pl.Int8))
        .sort("data_rank", "method", "metric", "task")
        .drop("data_rank")
    )

# %%
# 1. Format the AUC and CI columns into a single string
auc_df = metrics_df.filter(pl.col("metric") == "AUROC")

formatted_df = auc_df.with_columns(
    auc_with_ci=pl.when(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
    .then(
        pl.concat_str(
            [
                pl.col("value").round(3).cast(pl.String),
                pl.lit(" ["),
                pl.col("ci_lower").round(3).cast(pl.String),
                pl.lit(", "),
                pl.col("ci_upper").round(3).cast(pl.String),
                pl.lit("]"),
            ]
        )
    )
    .otherwise(pl.lit("NaN"))  # Handles the single-class fallback cases gracefully
)

# 2. Pivot the DataFrame
final_table = formatted_df.pivot(values="auc_with_ci", index=["data", "method"], on="task")

# Display the result
final_table.to_pandas().to_csv(OUTPUT_DIR / "auc_table.csv", index=False)
print(final_table)

# %%
plot_df = auc_df.filter(pl.col("task") == "MACRO").to_pandas()

# Calculate the distance from the median to the upper and lower bounds for the error bars
# Matplotlib expects error bars as [lower_errors, upper_errors] relative to the central point
yerr_lower = plot_df["value"] - plot_df["ci_lower"]
yerr_upper = plot_df["ci_upper"] - plot_df["value"]

plt.figure(figsize=(9, 6))

for label, (color, marker) in data_to_color_and_marker.items():
    mask = plot_df["data"] == label
    if mask.any():
        plt.errorbar(
            [textwrap.fill(method, 15) for method in plot_df.loc[mask, "method"]],
            plot_df.loc[mask, "value"],
            yerr=[yerr_lower[mask], yerr_upper[mask]],
            fmt=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1,
            color=color,  # Now receives a single string color per iteration
            ecolor="#444444",
            capsize=5,
            capthick=1.5,
            elinewidth=1.5,
            linestyle="None",
            label=label,  # Added label for a legend
        )
plt.axhline(0.5, color="red", linestyle="--", linewidth=2, label="Random")

plt.ylabel("ROC AUC")
plt.xlabel("Method")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="best")
plt.grid(axis="y", which="minor", linestyle=":", alpha=0.4)
plt.ylim(0, 1.0)


plt.tight_layout()
plt.savefig(OUTPUT_DIR / "macro_roc_auc.pdf", bbox_inches="tight")
plt.show()
