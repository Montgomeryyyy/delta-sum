import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def append_numeric_values_to_codes_expr(
    delimiter: str | None, numeric_values_sig_figs: int, code_col: str = "code", numeric_col: str = "numeric_value"
) -> pl.Expr:
    """Appends numeric values to codes with specified significant figures."""
    if not delimiter:
        raise ValueError("Delimiter must be specified to append numeric values to codes.")

    rounded_str = (
        pl.when(pl.col(numeric_col) == pl.col(numeric_col).floor())
        .then(pl.col(numeric_col).cast(pl.Int64).cast(pl.String))
        .otherwise(pl.col(numeric_col).round_sig_figs(numeric_values_sig_figs).cast(pl.String))
    )

    return (
        pl.when(pl.col(numeric_col).is_not_null())
        .then(pl.concat_str([pl.col(code_col), rounded_str], separator=delimiter))
        .otherwise(pl.col(code_col))
    )


def get_agg_expr(
    code_join_token: str,
    delimiter_replacement: str | None,
    delimiter: None | str,
    sort_by: str | None = None,
) -> pl.Expr:
    """Creates the standard Polars aggregation expression."""
    # We use a placeholder sort if needed, or just join
    expr = pl.col("code")

    # In the joined case, we need to sort; in the pretraining case,
    # we assume file order or lack of requirement.
    if sort_by is not None:
        expr = expr.sort_by(sort_by)

    expr = expr.str.join(code_join_token)
    if delimiter_replacement:
        if not delimiter:
            raise ValueError("Delimiter must be specified if delimiter_replacement is not None.")
        expr = expr.str.replace_all(delimiter, delimiter_replacement, literal=True)
    return expr


def meds_to_string_dataset(
    meds_dir: Path,
    output_dir: Path,
    overwrite: bool,
    splits: set[str] | None,
    code_join_token: str,
    delimiter: None | str,
    delimiter_replacement: str | None = None,
    append_numeric_values: bool = False,
    numeric_values_sig_figs: int = 2,
) -> None:
    # Format shards: split/shard.parquet
    shards = sorted(shard for shard in meds_dir.glob("*/*.parquet") if (splits is None or shard.parent.name in splits))
    agg_expr = get_agg_expr(code_join_token, delimiter_replacement, delimiter, sort_by="meds_row_idx")

    for shard in shards:
        split, name = shard.parent.name, shard.name
        out_path = output_dir / split / name

        if out_path.exists() and not overwrite:
            log.info(f"Skipping {out_path}: already exists.")
            continue

        try:
            log.info(f"Processing Split: [{split}] | Shard: {name}")

            # Simple scan and agg
            df = pl.scan_parquet(shard).with_row_index("meds_row_idx")
            if append_numeric_values:
                df = df.with_columns(
                    code=append_numeric_values_to_codes_expr(
                        delimiter=delimiter, numeric_values_sig_figs=numeric_values_sig_figs
                    )
                )

            df = df.group_by("subject_id").agg(ehr_text=agg_expr)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.sink_parquet(out_path)
            log.debug(f"Saved to {out_path}")

        except FileNotFoundError as e:
            log.warning(f"Skipping {shard}: {e}")
        except Exception as e:
            log.error(f"Error processing {shard}: {e}")
            raise


def meds_and_aces_to_string_dataset(
    meds_dir: Path,
    aces_dir: Path,
    output_dir: Path,
    overwrite: bool,
    tasks: set[str] | None,
    splits: set[str] | None,
    code_join_token: str,
    delimiter: None | str = None,
    delimiter_replacement: str | None = None,
    append_numeric_values: bool = False,
    numeric_values_sig_figs: int = 2,
) -> None:
    # Format shards: task/split/shard.parquet
    shards = sorted(
        shard
        for shard in aces_dir.glob("*/*/*.parquet")
        if (splits is None or shard.parent.name in splits) and (tasks is None or shard.parent.parent.name in tasks)
    )
    agg_expr = get_agg_expr(
        code_join_token,
        delimiter_replacement,
        delimiter,
        sort_by="meds_row_idx",
    )

    for shard in shards:
        task, split, name = shard.parent.parent.name, shard.parent.name, shard.name
        out_path = output_dir / task / split / name

        if out_path.exists() and not overwrite:
            log.info(f"Skipping {out_path}: already exists.")
            continue

        try:
            meds_p = meds_dir / split / name
            if not meds_p.exists():
                raise FileNotFoundError(f"MEDS shard missing: {meds_p}")

            log.info(f"Processing Task: [{task}] | Split: [{split}] | Shard: {name}")

            # Joined logic
            aces_lazy = pl.scan_parquet(shard)
            meds_lazy = pl.scan_parquet(meds_p).with_row_index("meds_row_idx")
            if append_numeric_values:
                meds_lazy = meds_lazy.with_columns(
                    code=append_numeric_values_to_codes_expr(
                        delimiter=delimiter, numeric_values_sig_figs=numeric_values_sig_figs
                    )
                )

            lf = (
                aces_lazy.join(meds_lazy, on="subject_id", how="inner")
                .filter(pl.col("time") <= pl.col("prediction_time"))
                .group_by(["subject_id", "prediction_time", "boolean_value"])
                .agg(ehr_text=agg_expr)
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            lf.sink_parquet(out_path)
            log.debug(f"Saved to {out_path}")

        except FileNotFoundError as e:
            log.warning(f"Skipping {shard}: {e}")
        except Exception as e:
            log.error(f"Error processing {shard}: {e}")
            raise
