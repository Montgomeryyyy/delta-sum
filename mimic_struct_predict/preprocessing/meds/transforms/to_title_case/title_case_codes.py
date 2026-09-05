#!/usr/bin/env python
"""Custom MAP stage that converts the code column to title case."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def title_case_codes(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that converts the code column to title case.

    Args:
        stage_cfg: Configuration for the stage (unused for this stage)

    Returns:
        Function that transforms a LazyFrame by title-casing the 'code' column.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [1, 2],
        ...     "code": ["abnormal", "normal"]
        ... }).lazy()
        >>> fn = title_case_codes(DictConfig({}))
        >>> result = fn(df).collect()
        >>> result.select("code")
        shape: (2, 1)
        ┌──────────┐
        │ code     │
        │ ---      │
        │ str      │
        ╞══════════╡
        │ Abnormal │
        │ Normal   │
        └──────────┘
    """

    def title_case_codes_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Convert 'code' column to title case."""
        return df.with_columns(pl.col("code").str.to_titlecase().alias("code"))

    return title_case_codes_fn
