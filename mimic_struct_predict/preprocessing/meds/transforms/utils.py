#!/usr/bin/env python
"""Shared utilities for MEDS transformation stages."""

import polars as pl


def _changed(col: str, fill_null: bool = True) -> pl.Expr:
    """Detect when a column value changes from the previous row."""
    return (pl.col(col) != pl.col(col).shift(1)).fill_null(fill_null)


def insert_parent_rows(
    df: pl.LazyFrame,
    change_cond: pl.Expr,
    code_expr: pl.Expr,
    extra_exprs: dict[str, pl.Expr] | None = None,
    exclude_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Inserts a parent row whenever the change condition is met.

    Args:
        df: Input LazyFrame
        change_cond: Boolean expression that triggers a new parent row
        code_expr: Expression for the 'code' column in the parent row
        extra_exprs: Dictionary mapping column names to expressions for the parent row.
                     If a column is not in this dict and is not 'subject_id', 'time', or 'code',
                     it will be null in the parent row.
        exclude_cols: Columns to exclude from the final output. These are typically
                      temporary columns used for calculations.
                      'subject_id', 'time', and 'code' are always included.

    Returns:
        LazyFrame with inserted parent rows
    """
    extra_exprs = extra_exprs or {}
    schema = df.collect_schema()

    # Ensure all extra columns exist in df
    for c in extra_exprs:
        if c not in schema.names():
            df = df.with_columns(pl.lit(None).alias(c))
    schema = df.collect_schema()

    # Define output columns (original order minus explicitly excluded ones)
    output_cols = [c for c in schema.names() if c not in (exclude_cols or [])]

    # Define parent row struct with matching field order
    parent_row_exprs = []
    for c in output_cols:
        if c == "subject_id":
            parent_row_exprs.append(pl.col("subject_id"))
        elif c == "time":
            parent_row_exprs.append(pl.col("time"))
        elif c == "code":
            parent_row_exprs.append(code_expr.alias("code"))
        elif c in extra_exprs:
            parent_row_exprs.append(extra_exprs[c].alias(c))
        else:
            parent_row_exprs.append(pl.lit(None, dtype=schema.get(c)).alias(c))

    parent_row = pl.struct(parent_row_exprs)

    # Define original row struct (matching output_cols order)
    actual_row = pl.struct(output_cols)

    return (
        df.with_columns(
            _rows=pl.when(change_cond)
            .then(pl.concat_list([parent_row, actual_row]))
            .otherwise(pl.concat_list([actual_row]))
        )
        .select("_rows")
        .explode("_rows")
        .unnest("_rows")
    )
