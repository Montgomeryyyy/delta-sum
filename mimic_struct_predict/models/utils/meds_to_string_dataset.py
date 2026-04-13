import logging
from pathlib import Path
from typing import Callable

import polars as pl

log = logging.getLogger(__name__)


def get_agg_expr(code_join_token: str, split_on_doubleslash: bool, sort_by: str | None = None) -> pl.Expr:
    """Creates the standard Polars aggregation expression."""
    # We use a placeholder sort if needed, or just join
    expr = pl.col("code")

    # In the joined case, we need to sort; in the pretraining case,
    # we assume file order or lack of requirement.
    if sort_by is not None:
        expr = expr.sort_by(sort_by)

    expr = expr.str.join(code_join_token)
    if split_on_doubleslash:
        expr = expr.str.replace_all("//", " ", literal=True)
    return expr


def run_processor_loop(
    input_paths: list[Path],
    output_dir: Path,
    overwrite: bool,
    process_fn: Callable[[Path], tuple[pl.LazyFrame, Path, str]],  # Returns (LazyFrame, OutPath, LogMsg)
) -> None:
    """Generic orchestrator for processing parquet shards."""
    for path in input_paths:
        try:
            lazy_df, out_path, context_msg = process_fn(path)

            if out_path.exists() and not overwrite:
                log.info(f"Skipping {out_path}: already exists.")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Processing {context_msg}")

            lazy_df.sink_parquet(out_path)
            log.debug(f"Saved to {out_path}")

        except FileNotFoundError as e:
            log.warning(f"Skipping {path}: {e}")
        except Exception as e:
            log.error(f"Error processing {path}: {e}")
            raise


## Refactored High-Level Functions


def meds_to_string_dataset(
    meds_dir: Path,
    output_dir: Path,
    splits: set[str],
    code_join_token: str,
    split_on_doubleslash: bool,
    overwrite: bool,
) -> None:
    # Format shards: split/shard.parquet
    shards = sorted(shard for shard in meds_dir.glob("*/*.parquet") if shard.parent.name in splits)
    agg_expr = get_agg_expr(code_join_token, split_on_doubleslash)

    def pretrain_loader(p: Path) -> tuple[pl.LazyFrame, Path, str]:
        split, name = p.parent.name, p.name
        out = output_dir / split / name
        # Simple scan and agg
        lf = pl.scan_parquet(p).group_by("subject_id").agg(ehr_text=agg_expr)
        return lf, out, f"Split: [{split}] | Shard: {name}"

    run_processor_loop(shards, output_dir, overwrite, pretrain_loader)


def meds_and_aces_to_string_dataset(
    meds_dir: Path,
    aces_dir: Path,
    output_dir: Path,
    tasks: set[str] | None,
    splits: set[str],
    code_join_token: str,
    split_on_doubleslash: bool,
    overwrite: bool,
) -> None:
    # Format shards: task/split/shard.parquet
    shards = sorted(
        shard
        for shard in aces_dir.glob("*/*/*.parquet")
        if shard.parent.name in splits and (tasks is None or shard.parent.parent.name in tasks)
    )
    agg_expr = get_agg_expr(code_join_token, split_on_doubleslash, sort_by="meds_row_idx")

    def task_loader(p: Path) -> tuple[pl.LazyFrame, Path, str]:
        task, split, name = p.parent.parent.name, p.parent.name, p.name
        meds_p = meds_dir / split / name
        if not meds_p.exists():
            raise FileNotFoundError(f"MEDS shard missing: {meds_p}")

        out = output_dir / task / split / name

        # Joined logic
        aces_lazy = pl.scan_parquet(p)
        meds_lazy = pl.scan_parquet(meds_p).with_row_index("meds_row_idx")

        lf = (
            aces_lazy.join(meds_lazy, on="subject_id", how="inner")
            .filter(pl.col("time") <= pl.col("prediction_time"))
            .group_by(["subject_id", "prediction_time", "boolean_value"])
            .agg(ehr_text=agg_expr)
        )
        return lf, out, f"Task: [{task}] | Split: [{split}] | Shard: {name}"

    run_processor_loop(shards, output_dir, overwrite, task_loader)
