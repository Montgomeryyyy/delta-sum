#!/usr/bin/env python
"""Shared utilities for MEDS transformation stages."""

import polars as pl


def _changed(expr: str | pl.Expr, fill_null: bool = True) -> pl.Expr:
    """Detect when an expression's value changes from the previous row."""
    if isinstance(expr, str):
        expr = pl.col(expr)
    return (expr != expr.shift(1)).fill_null(fill_null)


def insert_parent_rows(
    df: pl.LazyFrame,
    change_cond: pl.Expr,
    code_expr: pl.Expr,
    extra_exprs: dict[str, pl.Expr] | None = None,
) -> pl.LazyFrame:
    """Inserts a parent row whenever the change condition is met.

    Args:
        df: Input LazyFrame
        change_cond: Boolean expression that triggers a new parent row
        code_expr: Expression for the 'code' column in the parent row
        extra_exprs: Dictionary mapping column names to expressions for the parent row.
                     If a column is not in this dict and is not 'subject_id', 'time', or 'code',
                     it will be null in the parent row.

    Returns:
        LazyFrame with inserted parent rows
    """
    extra_exprs = extra_exprs or {}

    # We need to know the schema up front to ensure the parent rows we create
    # have the exact same shape (columns and types) as the original rows.
    schema = df.collect_schema()

    # Ensure all columns required for the parent rows exist in the dataframe.
    # If a new column is requested just for parent rows, we initialize it to
    # null in the original dataframe.
    for c in extra_exprs:
        if c not in schema.names():
            df = df.with_columns(pl.lit(None).alias(c))

    # Refresh the schema in case we just added new columns
    schema = df.collect_schema()

    # Define output columns
    output_cols = schema.names()

    # We attach an original row index so we can interleave parent and actual rows
    # without sorting by temporal / subject columns which could scramble events
    # occurring at the identical time.
    df = df.with_row_index("__row_idx")

    # Define the expressions to construct data for the parent rows.
    # We iterate over output_cols to ensure the columns are created in the
    # exact same order as the original dataframe.
    parent_row_exprs = []
    for c in output_cols:
        if c == "subject_id":
            # Parent rows share the same subject_id as the row that triggered them
            parent_row_exprs.append(pl.col("subject_id"))
        elif c == "time":
            # Parent rows share the exact same timestamp
            parent_row_exprs.append(pl.col("time"))
        elif c == "code":
            # Apply the logic that generates the parent's code
            parent_row_exprs.append(code_expr.alias("code"))
        elif c in extra_exprs:
            # Apply any custom expressions for the parent row
            parent_row_exprs.append(extra_exprs[c].alias(c))
        else:
            # Any remaining columns are empty (null) in the parent row.
            # We explicitly cast the null to match the original column's datatype
            parent_row_exprs.append(pl.lit(None, dtype=schema.get(c)).alias(c))

    # Construct the dataframe containing ONLY the newly generated parent rows.
    # We filter the original data where a parent row is needed, and generate new row values.
    # We include '__row_idx' and set '__is_actual' to 0 representing that this is a parent row.
    parents_df = df.filter(change_cond).select(
        *parent_row_exprs, pl.col("__row_idx"), pl.lit(0, dtype=pl.Int8).alias("__is_actual")
    )

    # Prepare the original dataframe to be identical in shape to parents_df.
    # We set '__is_actual' to 1 representing the original data rows.
    actual_df = df.select(*output_cols, pl.col("__row_idx"), pl.lit(1, dtype=pl.Int8).alias("__is_actual"))

    # Combine both DataFrames vertically.
    # We sort by __row_idx, then __is_actual.
    # For any given original row that triggered a parent row, both the new parent
    # and the original row will share the same __row_idx. Because parent has
    # __is_actual=0 and actual has __is_actual=1, the parent row will immediately
    # precede the actual row.
    return (
        pl.concat([parents_df, actual_df], how="vertical")
        .sort(["__row_idx", "__is_actual"])
        .drop(["__row_idx", "__is_actual"])
    )
