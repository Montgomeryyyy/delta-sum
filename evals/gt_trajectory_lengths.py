import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from evals.plot_style import (
    DEFAULT_FIG_HEIGHT_INCHES,
    NEURIPS_TEXTWIDTH_INCHES,
    SEPARATOR_LINE_KW,
    TIMELINE_PALETTE,
    apply_neurips_style,
    format_task_label,
    save_figure,
    sort_tasks,
)
from mimic_struct_predict.models.pre_training.generative_inference import detect_outcome, get_valid_categories
from mimic_struct_predict.models.utils.preprocessing import (
    append_numeric_values_to_codes_expr,
    get_agg_expr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Apply NeurIPS style across the module
apply_neurips_style()


def get_regex_patterns(cfg: DictConfig) -> tuple[re.Pattern, re.Pattern]:
    """Helper to compile the regex patterns exactly like generative_inference.py"""
    escaped_delimiter = re.escape(cfg.event_extraction.delimiter)
    escaped_date_prefix = re.escape(cfg.event_extraction.date_prefix)
    escaped_time_prefix = re.escape(cfg.event_extraction.time_prefix)

    date_pattern = re.compile(rf"^{escaped_date_prefix}\+(?P<days>\d+) days$")
    time_pattern = re.compile(rf"^{escaped_time_prefix}(?P<hours>\d+)h{escaped_delimiter}(?P<minutes>\d+)m$")
    return date_pattern, time_pattern


def get_future_text_df(aces_df: pl.DataFrame, meds_shard: Path, prep_cfg: DictConfig) -> pl.DataFrame:
    """Joins ACES with MEDS and filters for events after prediction_time."""
    meds_lazy = pl.scan_parquet(meds_shard).with_row_index("meds_row_idx")

    if prep_cfg.pre_processing.append_numeric_values:
        meds_lazy = meds_lazy.with_columns(
            code=append_numeric_values_to_codes_expr(
                delimiter=prep_cfg.pre_processing.delimiter,
                numeric_values_sig_figs=prep_cfg.pre_processing.numeric_values_sig_figs,
            )
        )

    agg_expr = get_agg_expr(
        code_join_token=prep_cfg.pre_processing.code_join_token,
        delimiter_replacement=prep_cfg.pre_processing.delimiter_replacement,
        delimiter=prep_cfg.pre_processing.delimiter,
        sort_by="meds_row_idx",
    )

    lf = (
        aces_df.lazy()
        .join(meds_lazy, on="subject_id", how="inner")
        .filter(pl.col("time") > pl.col("prediction_time"))
        .group_by(["subject_id", "prediction_time", "boolean_value"])
        .agg(ehr_text=agg_expr)
    )
    return lf.collect()


def get_initial_categories(
    aces_df: pl.DataFrame, meds_shard: Path, prep_cfg: DictConfig, category_prefix: str
) -> dict[tuple[int, Any], str]:
    """Finds the last category prior to prediction_time for each subject_id directly from the dataframe."""
    meds_lazy = pl.scan_parquet(meds_shard).with_row_index("meds_row_idx")

    code_expr = pl.col("code")
    if prep_cfg.pre_processing.delimiter_replacement:
        code_expr = code_expr.str.replace_all(
            prep_cfg.pre_processing.delimiter, prep_cfg.pre_processing.delimiter_replacement, literal=True
        )

    lf = (
        aces_df.lazy()
        .join(meds_lazy, on="subject_id", how="inner")
        .filter(pl.col("time") <= pl.col("prediction_time"))
        .with_columns(code=code_expr)
        .filter(pl.col("code").str.starts_with(category_prefix))
        .group_by(["subject_id", "prediction_time"], maintain_order=True)
        .agg(pl.col("code").sort_by("meds_row_idx").last().alias("initial_category"))
    )

    df = lf.collect()
    if df.height == 0:
        return {}
    return dict(zip(zip(df["subject_id"].to_list(), df["prediction_time"].to_list()), df["initial_category"].to_list()))


def get_token_count_for_modality(
    future_df: pl.DataFrame,
    initial_categories: dict[tuple[int, Any], str],
    cfg: DictConfig,
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    valid_categories: set[str],
    date_pattern: re.Pattern,
    time_pattern: re.Pattern,
) -> dict[tuple[int, Any], int]:
    """Runs detect_outcome to slice ground truth and compute token length."""
    task_config = cfg.event_extraction.tasks[task_name]
    target_event_regex = re.compile(task_config.target_event_regex)
    end_event_str = task_config.end_event_regex
    end_event_regex = re.compile(end_event_str) if end_event_str is not None else None

    code_splitter = cfg.event_extraction.code_splitter
    patient_tokens = {}
    logged_example = False

    for row in future_df.iter_rows(named=True):
        subj = row["subject_id"]
        text = row["ehr_text"]
        pred_time = row["prediction_time"]
        if not text:
            continue

        initial_category = initial_categories.get((subj, pred_time), "")
        initial_time = pred_time.hour / 24.0 + pred_time.minute / 1440.0

        result = detect_outcome(
            text=text,
            code_splitter=code_splitter,
            delimiter=cfg.event_extraction.delimiter,
            category_prefix=cfg.event_extraction.category_prefix,
            time_prefix=cfg.event_extraction.time_prefix,
            date_prefix=cfg.event_extraction.date_prefix,
            date_pattern=date_pattern,
            time_pattern=time_pattern,
            target_event_regex=target_event_regex,
            end_event_regex=end_event_regex,
            end_time_days=task_config.end_time_days,
            initial_time_offset=initial_time,
            initial_category=initial_category,
            valid_categories=valid_categories,
            max_day_gap=cfg.event_extraction.max_day_gap,
        )

        # Use the new num_lines field to precisely slice the string up to the exit event!
        num_lines = result["num_lines"]
        lines = [line.strip() for line in text.split(code_splitter) if line.strip()]
        sliced_text = code_splitter.join(lines[:num_lines])

        patient_tokens[(subj, pred_time)] = len(tokenizer.encode(sliced_text, add_special_tokens=False))

        if not logged_example:
            log.info(f"--- Example for {task_name} ---")
            log.info(f"Initial Category: {initial_category}")
            log.info(f"Initial Time: {initial_time}")
            log.info(f"Original Text (first 1000 chars): {text[:1000]}...")
            log.info(f"Detect Outcome Result: {result}")
            log.info(f"Sliced Text: {sliced_text}...")
            log.info(f"Token Count: {patient_tokens[(subj, pred_time)]}")
            log.info("-" * 40)
            logged_example = True

    return patient_tokens


def process_shard(
    aces_shard: Path,
    nl_meds_shard: Path,
    struct_meds_shard: Path,
    nl_cfg: DictConfig,
    struct_cfg: DictConfig,
    nl_prep_cfg: DictConfig,
    struct_prep_cfg: DictConfig,
    nl_tokenizer: PreTrainedTokenizerBase,
    struct_tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    nl_valid_cats: set[str],
    struct_valid_cats: set[str],
) -> pl.DataFrame | None:
    """Processes a single ACES shard for both modalities."""
    if not aces_shard.exists():
        log.warning(f"ACES shard not found: {aces_shard}")
        return None

    aces_df = pl.read_parquet(aces_shard)
    if aces_df.height == 0:
        log.warning(f"ACES shard is empty: {aces_shard}")
        return None

    records = []
    nl_date, nl_time = get_regex_patterns(nl_cfg)
    struct_date, struct_time = get_regex_patterns(struct_cfg)

    nl_tokens = {}
    nl_initial_cats = {}
    if not nl_meds_shard.exists():
        log.warning(f"NL MEDS shard not found: {nl_meds_shard}")
    else:
        nl_initial_cats = get_initial_categories(
            aces_df, nl_meds_shard, nl_prep_cfg, nl_cfg.event_extraction.category_prefix
        )
        nl_future_df = get_future_text_df(aces_df, nl_meds_shard, nl_prep_cfg)
        nl_tokens = get_token_count_for_modality(
            nl_future_df, nl_initial_cats, nl_cfg, task_name, nl_tokenizer, nl_valid_cats, nl_date, nl_time
        )

    struct_tokens = {}
    struct_initial_cats = {}
    if not struct_meds_shard.exists():
        log.warning(f"Structured MEDS shard not found: {struct_meds_shard}")
    else:
        struct_initial_cats = get_initial_categories(
            aces_df, struct_meds_shard, struct_prep_cfg, struct_cfg.event_extraction.category_prefix
        )
        struct_future_df = get_future_text_df(aces_df, struct_meds_shard, struct_prep_cfg)
        struct_tokens = get_token_count_for_modality(
            struct_future_df,
            struct_initial_cats,
            struct_cfg,
            task_name,
            struct_tokenizer,
            struct_valid_cats,
            struct_date,
            struct_time,
        )

    for row in aces_df.iter_rows(named=True):
        subj = row["subject_id"]
        pred_time = row["prediction_time"]
        records.append(
            {
                "task": task_name,
                "subject_id": subj,
                "prediction_time": pred_time,
                "boolean_value": row["boolean_value"],
                "nl_tokens": nl_tokens.get((subj, pred_time), 0),
                "struct_tokens": struct_tokens.get((subj, pred_time), 0),
            }
        )

    if not records:
        return None
    return pl.DataFrame(records)


def plot_trajectory_lengths(
    df: pl.DataFrame,
    tasks: list[str],
    max_tokens: int,
    out_dir: Path,
):
    """Plots ground truth trajectory length distributions for Structured and Natural Language."""
    tasks = sort_tasks(tasks)
    plot_df = (
        df.filter(pl.col("task").is_in(tasks))
        .select(["task", "struct_tokens", "nl_tokens"])
        .unpivot(index=["task"], variable_name="modality", value_name="tokens")
        .filter((pl.col("tokens").is_not_null()) & (pl.col("tokens") > 0))
        .with_columns(
            pl.col("modality").replace({"struct_tokens": "Structured (GT)", "nl_tokens": "Natural Language (GT)"}),
            pl.col("task").replace({t: format_task_label(t) for t in tasks}),
        )
    )

    task_labels = [format_task_label(t) for t in tasks]
    fig, ax = plt.subplots(figsize=(NEURIPS_TEXTWIDTH_INCHES, DEFAULT_FIG_HEIGHT_INCHES))

    sns.boxplot(
        data=plot_df.to_pandas(),
        x="task",
        y="tokens",
        hue="modality",
        order=task_labels,
        hue_order=["Structured (GT)", "Natural Language (GT)"],
        palette=TIMELINE_PALETTE,
        saturation=1,
        linewidth=0.6,
        whis=[5, 95],
        gap=0.1,
        flierprops=dict(marker="o", markersize=0.8, alpha=0.15, markeredgewidth=0, markerfacecolor="#333333"),
        ax=ax,
    )

    # Apply diagonal hatching to Natural Language (GT) boxes for cross-figure texture consistency
    nl_rgba = mcolors.to_rgba(TIMELINE_PALETTE["Natural Language (GT)"])
    for patch in ax.patches:
        patch_fc = patch.get_facecolor()
        if np.allclose(patch_fc[:3], nl_rgba[:3], atol=1e-2):
            patch.set_hatch("///")
            patch.set_edgecolor("#222222")
            patch.set_linewidth(0.6)

    for artist in ax.artists:
        artist_fc = artist.get_facecolor()
        if np.allclose(artist_fc[:3], nl_rgba[:3], atol=1e-2):
            artist.set_hatch("///")
            artist.set_edgecolor("#222222")
            artist.set_linewidth(0.6)

    for sep_x in range(len(task_labels) - 1):
        ax.axvline(sep_x + 0.5, **SEPARATOR_LINE_KW)

    ax.axhline(
        y=max_tokens,
        color=TIMELINE_PALETTE["threshold_line"],
        linestyle="--",
        linewidth=1.2,
        label=f"Our Max New Tokens ({max_tokens})",
        zorder=3,
    )

    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(bottom=1)
    ax.set_xlabel("")
    ax.set_ylabel("Ground Truth Trajectory Length (Tokens)")
    sns.despine(ax=ax, trim=False)

    legend_handles = [
        Patch(
            facecolor=TIMELINE_PALETTE["Structured (GT)"],
            edgecolor="#222222",
            linewidth=0.5,
            label="Structured (GT)",
        ),
        Patch(
            facecolor=TIMELINE_PALETTE["Natural Language (GT)"],
            hatch="///",
            edgecolor="#222222",
            linewidth=0.5,
            label="Natural Language (GT)",
        ),
        Line2D(
            [0],
            [0],
            color=TIMELINE_PALETTE["threshold_line"],
            linestyle="--",
            linewidth=1.2,
            label=f"Our Max New Tokens ({max_tokens})",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
    )

    plt.tight_layout()

    stem = "token_lengths_grouped_boxplot"
    save_figure(out_dir, stem)
    plt.close()
    log.info(f"Saved camera-ready token length plots to {out_dir / f'{stem}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Extract token counts up to exit event for generative sequences.")
    parser.add_argument("--tasks", nargs="+", default=os.environ.get("ACES_TASKS").split())
    parser.add_argument("--split", type=str, default="tuning", choices=["train", "tuning", "held_out"])

    parser.add_argument(
        "--aces_dir", type=str, default=os.environ.get("ACES_OUTPUT_DIR"), help="ACES labels output directory"
    )
    parser.add_argument(
        "--nl_meds_dir",
        type=str,
        default=os.environ.get("NATURAL_LANGUAGE_MEDS_DIR"),
        help="NL MEDS directory (e.g. add_delta_rows)",
    )
    parser.add_argument(
        "--struct_meds_dir",
        type=str,
        default=os.environ.get("STRUCTURED_MEDS_DIR"),
        help="Structured MEDS directory (e.g. data)",
    )

    parser.add_argument(
        "--nl_config",
        type=str,
        default="mimic_struct_predict/models/pre_training/config/generative_inference/natural_language.yaml",
        help="Path to natural_language.yaml",
    )
    parser.add_argument(
        "--struct_config",
        type=str,
        default="mimic_struct_predict/models/pre_training/config/generative_inference/structured.yaml",
        help="Path to structured.yaml",
    )

    parser.add_argument(
        "--nl_prep_config",
        type=str,
        default="mimic_struct_predict/models/embedding_and_lr/config/create_task_datasets_nl.yaml",
        help="Path to create_task_datasets_nl.yaml",
    )
    parser.add_argument(
        "--struct_prep_config",
        type=str,
        default="mimic_struct_predict/models/embedding_and_lr/config/create_task_datasets_structured.yaml",
        help="Path to create_task_datasets_structured.yaml",
    )

    parser.add_argument("--nl_tokenizer_path", type=str, required=True)
    parser.add_argument("--struct_tokenizer_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default=f"{os.environ.get('EVAL_PLOTS_DIR')}/gt_trajectory_lengths")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Token limit for the plot")
    args = parser.parse_args()
    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    aces_base = Path(args.aces_dir)
    nl_meds_base = Path(args.nl_meds_dir)
    struct_meds_base = Path(args.struct_meds_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading configs and tokenizers...")
    nl_cfg = OmegaConf.load(args.nl_config)
    log.info(nl_cfg)
    struct_cfg = OmegaConf.merge(nl_cfg, OmegaConf.load(args.struct_config))
    log.info(struct_cfg)

    nl_prep_cfg = OmegaConf.load(args.nl_prep_config)
    struct_prep_cfg = OmegaConf.merge(nl_prep_cfg, OmegaConf.load(args.struct_prep_config))

    nl_tokenizer = AutoTokenizer.from_pretrained(args.nl_tokenizer_path)
    struct_tokenizer = AutoTokenizer.from_pretrained(args.struct_tokenizer_path)

    # Note: If running from repo root, this relative path from the config works.
    nl_valid_cats = get_valid_categories(nl_cfg.event_extraction.event_config_path)
    struct_valid_cats = get_valid_categories(struct_cfg.event_extraction.event_config_path)

    all_dfs = []
    for task_name in args.tasks:
        if task_name not in nl_cfg.event_extraction.tasks:
            log.warning(f"Task {task_name} not found in NL config. Skipping.")
            continue

        aces_split_dir = aces_base / task_name / args.split

        if not aces_split_dir.exists():
            log.warning(f"Split {args.split} not found for task {task_name}. Skipping.")
            continue

        aces_shards = sorted(aces_split_dir.glob("*.parquet"))
        log.info(f"Processing Task: [{task_name}] | Split: [{args.split}] | Shards: {len(aces_shards)}")

        for aces_shard in aces_shards:
            log.info(f"Processing shard: {aces_shard}")
            nl_meds_shard = nl_meds_base / args.split / aces_shard.name
            struct_meds_shard = struct_meds_base / args.split / aces_shard.name

            df_shard = process_shard(
                aces_shard,
                nl_meds_shard,
                struct_meds_shard,
                nl_cfg,
                struct_cfg,
                nl_prep_cfg,
                struct_prep_cfg,
                nl_tokenizer,
                struct_tokenizer,
                task_name,
                nl_valid_cats,
                struct_valid_cats,
            )
            if df_shard is not None:
                all_dfs.append(df_shard)

    if not all_dfs:
        log.warning("No records found to process.")
        return

    df = pl.concat(all_dfs)

    parquet_out = out_dir / "gt_trajectory_lengths.parquet"
    df.write_parquet(parquet_out)
    log.info(f"Saved raw metrics to {parquet_out}")

    # Load previously saved metrics if data generation is commented out
    parquet_out = out_dir / "gt_trajectory_lengths.parquet"
    if "df" not in locals():
        if parquet_out.exists():
            df = pl.read_parquet(parquet_out)
        else:
            log.error(f"Dataframe 'df' not defined and {parquet_out} not found. Please uncomment data generation.")
            return

    plot_trajectory_lengths(df, args.tasks, args.max_tokens, out_dir)


if __name__ == "__main__":
    main()
