import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.patches import Patch

from mimic_struct_predict.evals.plot_style import (
    DEFAULT_FIG_HEIGHT_INCHES,
    ERROR_RATES_PALETTE,
    MODALITY_STYLES,
    NEURIPS_TEXTWIDTH_INCHES,
    SEPARATOR_LINE_KW,
    TASK_MAP,
    apply_neurips_style,
    format_task_title,
    get_modality_legend_handles,
    save_figure,
    sort_tasks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

apply_neurips_style()


def load_predictions(base_dir: Path, task: str, split: str) -> pl.DataFrame:
    """Loads prediction parquets for Rollout models."""
    task_dir = base_dir / split / task
    latest = max(d for d in task_dir.iterdir() if d.is_dir())
    log.info(f"Loading predictions from {latest}")
    return pl.read_parquet(list(latest.glob("*.parquet")))


def format_error_pct(val: float) -> str:
    """Formats percentage values to 2 decimal places, bucketing small errors (< 0.01%) as '$<0.01$'."""
    if val < 0.01:
        return "$<0.01$"
    return f"{val:.2f}"


def save_latex_table(df: pl.DataFrame, out_path: Path) -> None:
    """Generates a camera-ready LaTeX table using pandas DataFrame.to_latex()."""
    pdf = df.to_pandas().rename(columns=lambda c: c.replace("%", r"\%"))
    numeric_cols = [c for c in pdf.columns if c not in ["Modality", "Task"]]

    latex_table = pdf.to_latex(
        index=False,
        escape=False,
        formatters={c: format_error_pct for c in numeric_cols},
    )
    out_path.write_text(latex_table)
    log.info(f"Saved LaTeX table to {out_path}")


def save_error_samples(all_modality_errors: dict[str, pl.DataFrame], out_path: Path, n: int = 10) -> None:
    """Outputs randomly selected error lines for each error type across modalities to file."""
    lines = ["# Rollout Error Samples\n"]

    for mod_name, df_errors in all_modality_errors.items():
        lines.append(f"## {mod_name}\n")
        for reason in ERROR_RATES_PALETTE:
            sub = df_errors.filter(pl.col("finish_reason") == reason)
            if sub.is_empty():
                continue
            lines.append(f"### {reason}")
            samples = sub.sample(n=min(n, sub.height), seed=42)["error_line"].to_list()
            lines.extend([f"{i}. `{err}`" for i, err in enumerate(samples, 1)] + [""])

    out_path.write_text("\n".join(lines).strip() + "\n")
    log.info(f"Saved error samples to {out_path}")


def plot_error_rates(
    all_modality_stats: dict[str, dict[str, dict[str, float]]],
    tasks: list[str],
    out_dir: Path,
) -> None:
    """Plots format error rates grouped by task and stacked by finish reason with publication styling."""
    modalities = list(all_modality_stats.keys())
    n_mods = len(modalities)
    x_positions = np.arange(len(tasks))
    bar_width = 0.35

    all_reasons = {r for mod in all_modality_stats.values() for t in tasks for r in mod.get(t, {})}
    unknown_reasons = all_reasons - ERROR_RATES_PALETTE.keys()
    if unknown_reasons:
        raise ValueError(
            f"Encountered unexpected finish reasons: {unknown_reasons}. "
            f"Allowed reasons must be defined in ERROR_RATES_PALETTE: {list(ERROR_RATES_PALETTE.keys())}"
        )

    ordered_reasons = [r for r in ERROR_RATES_PALETTE if r in all_reasons]

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXTWIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES))
    y_max = 0.05

    for mod_idx, mod_name in enumerate(modalities):
        style = MODALITY_STYLES[mod_name]
        mod_offset = (mod_idx - (n_mods - 1) / 2) * bar_width
        bar_x = x_positions + mod_offset
        bottoms = np.zeros(len(tasks))

        for reason in ordered_reasons:
            props = [all_modality_stats[mod_name][task].get(reason, 0.0) for task in tasks]
            ax.bar(
                bar_x,
                props,
                bar_width * 0.92,
                bottom=bottoms,
                color=ERROR_RATES_PALETTE[reason],
                hatch=style["hatch"],
                alpha=style["alpha"],
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
            bottoms += np.array(props)

        if len(bottoms) > 0:
            y_max = max(y_max, float(np.max(bottoms)))

    # Subtle separator lines between task clusters
    for sep_x in x_positions[:-1] + 0.5:
        ax.axvline(sep_x, **SEPARATOR_LINE_KW)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Trajectory Error Rate")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([TASK_MAP[t] for t in tasks], rotation=0, ha="center")
    ax.tick_params(axis="x", which="major", pad=12)

    minor_xticks = [pos + (mod_idx - (n_mods - 1) / 2) * bar_width for pos in x_positions for mod_idx in range(n_mods)]
    minor_xlabels = [MODALITY_STYLES[mod]["prefix"] for _ in x_positions for mod in modalities]
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_xticklabels(minor_xlabels, minor=True)
    ax.tick_params(axis="x", which="minor", length=0, pad=2)

    ax.set_ylim(0, max(0.05, y_max * 1.15))

    # Clean 2-row legend: Row 1 = Error types, Row 2 = Modalities (S vs NL)
    reason_handles = [
        Patch(facecolor=ERROR_RATES_PALETTE[r], edgecolor="#222222", linewidth=0.5, label=r) for r in ordered_reasons
    ]
    modality_handles = get_modality_legend_handles(facecolor="white")
    legend_handles = reason_handles + modality_handles

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
    )

    sns.despine(ax=ax, trim=False)
    plt.tight_layout()

    stem = "combined_error_rates"
    save_figure(out_dir, stem)
    plt.close()
    log.info(f"Saved error plot to {out_dir / f'{stem}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rollout format adherence and error rates.")
    parser.add_argument("--split", type=str, default="held_out", choices=["train", "tuning", "held_out"])
    parser.add_argument("--tasks", nargs="+", default=os.environ.get("ACES_TASKS", "").split())
    parser.add_argument(
        "--struct_dir",
        type=Path,
        default=Path(os.environ.get("STRUCTURED_INFERENCE_DIR", "")) / "pt" / "rollout",
    )
    parser.add_argument(
        "--nl_dir",
        type=Path,
        default=Path(os.environ.get("NATURAL_LANGUAGE_INFERENCE_DIR", "")) / "cpt" / "rollout",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "error_rates",
    )
    args = parser.parse_args()
    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"Structured Rollout": args.struct_dir, "Natural Language Rollout": args.nl_dir}

    summary_rows = []
    all_modality_stats = {}
    all_modality_errors = {}

    for mod_name, mod_dir in sources.items():
        log.info(f"--- Evaluating {mod_name} ---")
        task_stats = {}
        mod_errors = []

        for task in args.tasks:
            roll_df = load_predictions(mod_dir, task, args.split)
            exploded = (
                roll_df.select(["format_adherence", "finish_reason", "error_line"])
                .explode(["format_adherence", "finish_reason", "error_line"])
                .with_columns(pl.col("finish_reason").replace({"Time Backward Error": "Time Error"}))
            )

            n_total = exploded.height
            errors_df = exploded.filter(~pl.col("format_adherence"))
            n_errors = errors_df.height

            if n_errors > 0:
                mod_errors.append(errors_df.select(["finish_reason", "error_line"]))

            reason_props = (
                {r: c / n_total for r, c in errors_df["finish_reason"].value_counts().iter_rows()}
                if n_errors > 0
                else {}
            )

            summary_rows.append(
                {
                    "Modality": mod_name,
                    "Task": format_task_title(task),
                    "Date Error (%)": reason_props.get("Date Error", 0.0) * 100,
                    "Time Error (%)": reason_props.get("Time Error", 0.0) * 100,
                    "Category Error (%)": reason_props.get("Category Error", 0.0) * 100,
                    "Max New Token Reached (%)": reason_props.get("Max New Tokens Reached", 0.0) * 100,
                    "Total Error (%)": (n_errors / n_total * 100) if n_total > 0 else 0.0,
                }
            )

            task_stats[task] = reason_props
            log.info(f"Task '{task}': {n_errors}/{n_total} errors ({(n_errors/n_total*100):.2f}%)")

        all_modality_stats[mod_name] = task_stats
        if mod_errors:
            all_modality_errors[mod_name] = pl.concat(mod_errors)

    plot_error_rates(all_modality_stats, args.tasks, args.output_dir)

    df = pl.DataFrame(summary_rows)
    csv_path = args.output_dir / "rollout_error_rates.csv"
    tex_path = args.output_dir / "rollout_error_rates.tex"
    md_path = args.output_dir / "rollout_error_rates.md"
    df.write_csv(csv_path)
    save_latex_table(df, tex_path)
    md_path.write_text(df.to_pandas().to_markdown(index=False, floatfmt=".2f"))
    log.info(f"Saved summary tables to {csv_path}")

    save_error_samples(all_modality_errors, args.output_dir / "rollout_error_samples.md")


if __name__ == "__main__":
    main()
