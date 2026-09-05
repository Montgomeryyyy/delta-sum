#!/usr/bin/env python
"""Custom MAP stage that replaces regex patterns in a string column."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig


@Stage.register(is_metadata=False)
def regex_replacement(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that performs regex replace on a string column.

    Args:
        stage_cfg: Configuration containing:
            - column: Column name to clean (default: "code")
            - patterns: List of replacement patterns, each with 'pattern' (regex) and
              'replacement' keys

    Returns:
        Function that transforms a LazyFrame by applying regex replacements
    """

    patterns = stage_cfg.get("patterns", [])

    def regex_replacement_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply regex replacements to the specified column."""
        cleaned_col = pl.col(stage_cfg.column)

        for pattern_cfg in patterns:
            pattern = pattern_cfg["pattern"]
            replacement = pattern_cfg["replacement"]
            # literal=False enforces that this pattern is evaluated as regex
            cleaned_col = cleaned_col.str.replace_all(pattern, replacement, literal=False)

        return df.with_columns(cleaned_col.alias(stage_cfg.column))

    return regex_replacement_fn
