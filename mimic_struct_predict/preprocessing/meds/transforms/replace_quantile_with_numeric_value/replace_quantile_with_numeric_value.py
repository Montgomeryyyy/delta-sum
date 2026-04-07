#!/usr/bin/env python
"""Custom MAP stage that replaces quantile suffixes with numeric values."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def replace_quantile_with_numeric_value(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that replaces quantile suffixes with numeric values.

    Args:
        stage_cfg: Configuration for the stage.
            - drop_numeric_values: If True, set numeric_value to null and strip quantile
              from code without appending numeric value.

    Returns:
        Function that transforms a LazyFrame.
    """
    drop_numeric_values = stage_cfg.get("drop_numeric_values", False)
    sig_figs = stage_cfg.get("sig_figs", 3)

    def replace_quantile_with_numeric_value_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Replace quantile suffix with numeric value or drop it."""
        # Convert numeric_value to string with appropriate formatting
        # Ints stay as ints, floats are rounded to sig figs
        rounded_str = (
            pl.when(pl.col("numeric_value") == pl.col("numeric_value").floor())
            .then(pl.col("numeric_value").cast(pl.Int64).cast(pl.String))
            .otherwise(pl.col("numeric_value").round_sig_figs(sig_figs).cast(pl.String))
        )
        # Remove everything after final // (if any) and append rounded numeric value
        df = df.with_columns(
            code=pl.when(pl.col("numeric_value").is_null())
            .then(pl.col("code"))
            .otherwise(pl.col("code").str.replace(r"//[^//]*$", "") + "//" + rounded_str)
        )

        if drop_numeric_values:
            df = df.drop("numeric_value")

        return df

    return replace_quantile_with_numeric_value_fn
