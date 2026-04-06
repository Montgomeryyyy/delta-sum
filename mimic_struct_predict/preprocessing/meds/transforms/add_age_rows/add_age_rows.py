#!/usr/bin/env python
"""Custom MAP stage that adds a row with patient age when subjects or dates change."""

from collections.abc import Callable

import polars as pl
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig

from ..utils import _changed, insert_parent_rows


@Stage.register(is_metadata=False)
def add_age_rows(
    stage_cfg: DictConfig,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that inserts a row with "Age Y years//N days" on changes.

    The row is inserted whenever the subject_id or the date of the event changes.
    Age is calculated relative to a birth event identified by a specific prefix.

    Args:
        stage_cfg: Configuration for the stage containing:
            - birth_prefix: Prefix identifying birth events (default: "Birth")

    Returns:
        Function that transforms a LazyFrame by inserting age rows

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1],
        ...     "time": [
        ...         datetime(2000, 1, 1),
        ...         datetime(2020, 6, 15),
        ...         datetime(2020, 6, 16),
        ...     ],
        ...     "code": ["Born", "Lab Test", "Diagnosis"],
        ...     "prefix": ["Birth", "Lab", "Clinical"],
        ... }).lazy()
        >>> fn = add_age_rows(DictConfig({"birth_prefix": "Birth"}))
        >>> result = fn(df).collect()
        >>> result.select("subject_id", "time", "code")
        shape: (6, 3)
        ┌────────────┬─────────────────────┬───────────────────┐
        │ subject_id ┆ time                ┆ code              │
        │ ---        ┆ ---                 ┆ ---               │
        │ i64        ┆ datetime[μs]        ┆ str               │
        ╞════════════╪═════════════════════╪═══════════════════╡
        │ 1          ┆ 2000-01-01 00:00:00 ┆ Age 0 years//0 days│
        │ 1          ┆ 2000-01-01 00:00:00 ┆ Born              │
        │ 1          ┆ 2020-06-15 00:00:00 ┆ Age 20 years//166 days│
        │ 1          ┆ 2020-06-15 00:00:00 ┆ Lab Test          │
        │ 1          ┆ 2020-06-16 00:00:00 ┆ Age 20 years//167 days│
        │ 1          ┆ 2020-06-16 00:00:00 ┆ Diagnosis         │
        └────────────┴─────────────────────┴───────────────────┘
    """

    header = stage_cfg.get("header", "")

    def add_age_rows_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        """Insert Age rows on subject or date changes."""

        # 1. Calculate Birth Time for each subject
        birth_time = pl.col("time").filter(pl.col("code") == stage_cfg.birth_code).first().over("subject_id")

        # 2. Calculate age_year and age_day (logic from calculate_age.py)
        current_year = pl.col("time").dt.year()
        current_month = pl.col("time").dt.month()
        current_day = pl.col("time").dt.day()

        birth_year = birth_time.dt.year()
        birth_month = birth_time.dt.month()
        birth_day = birth_time.dt.day()

        # Age in years
        age_years = (
            current_year
            - birth_year
            - pl.when((current_month < birth_month) | ((current_month == birth_month) & (current_day < birth_day)))
            .then(1)
            .otherwise(0)
        )

        # Last birthday date and days since then
        last_birthday = birth_time.dt.offset_by(pl.format("{}y", age_years))
        age_days = (pl.col("time") - last_birthday).dt.total_days()

        # 3. Format the age string
        # Formatted as "Age Y years//N days"
        df = df.with_columns(
            _formatted_age=pl.lit(header) + pl.format("Age {} years//{} days", age_years, age_days),
            _date=pl.col("time").dt.date(),
        )

        # 4. Detect when subject or date changes
        change_cond = _changed("subject_id") | _changed("_date")

        # 5. Use utility to insert parent rows
        return insert_parent_rows(
            df,
            change_cond=change_cond,
            code_expr=pl.col("_formatted_age"),
            exclude_cols=["_formatted_age", "_date"],
        )

    return add_age_rows_fn
