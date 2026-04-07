#!/usr/bin/env python
"""Custom MAP stage that merges age code with text value and nulls numeric value."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def merge_age_column(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that merges the age code with the text value column.

    For rows where the code matches `age_code`, the code is updated to
    `code + "//" + age_text_value` and the `numeric_value` is set to null.

    Args:
        stage_cfg: Configuration for the stage containing:
            - age_code: The code to match for merging (default: "Age")
            - drop_age_text_column: Whether to drop the age_text_value column after merging (default: True)

    Returns:
        Function that transforms a LazyFrame by merging age columns
    """

    age_code = stage_cfg.get("age_code", "Age")
    drop_age_text_column = stage_cfg.get("drop_age_text_column", True)

    def merge_age_column_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Merge age code with text value and null numeric value."""

        is_age = pl.col("code") == age_code
        has_text = pl.col("age_text_value").is_not_null()

        result = df.with_columns(
            code=pl.when(is_age & has_text)
            .then(pl.col("code") + pl.lit("//") + pl.col("age_text_value"))
            .otherwise(pl.col("code")),
            numeric_value=pl.when(is_age).then(None).otherwise(pl.col("numeric_value")),
        )

        if drop_age_text_column:
            result = result.drop("age_text_value")

        return result

    return merge_age_column_fn
