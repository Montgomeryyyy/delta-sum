import polars as pl
df = pl.DataFrame({"numeric_value": [34, 34.0, 34.121, 0.001234]})
res = df.with_columns(
    rounded_str=pl.when(pl.col("numeric_value") == pl.col("numeric_value").round(0))
        .then(pl.col("numeric_value").cast(pl.Int64).cast(pl.String))
        .otherwise(pl.col("numeric_value").round_sig_figs(3).cast(pl.String))
)
print(res)
