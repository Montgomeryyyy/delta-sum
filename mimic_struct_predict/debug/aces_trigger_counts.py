"""Trace where ACES loses trigger events for one MEDS shard.

Example:
    uv run python mimic_struct_predict/debug/aces_trigger_counts.py \
        --config mimic_struct_predict/preprocessing/aces_task_definitions/ptb.yaml \
        --shard ${MEDS_COHORT_DIR}/data/train/0.parquet
"""

import argparse
from pathlib import Path

import polars as pl
from aces.config import TaskExtractorConfig
from aces.predicates import get_predicates_df
from aces.query import query
from omegaconf import OmegaConf


def _summary(df: pl.DataFrame) -> str:
    return f"{df.height:,} rows, {df['subject_id'].n_unique() if df.height else 0:,} subjects"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    args = parser.parse_args()

    cfg = TaskExtractorConfig.load(config_path=args.config)
    trigger = cfg.trigger.predicate
    regex = cfg.predicates[trigger].code["regex"]
    print(f"Trigger '{trigger}', regex {regex!r}")

    for name, use_pyarrow in [("polars reader", False), ("pyarrow reader (used by ACES)", True)]:
        raw = pl.read_parquet(args.shard, use_pyarrow=use_pyarrow)
        hits = raw.filter(pl.col("code").cast(pl.Utf8).str.contains(regex))
        print(f"{name}: {_summary(hits)} match, of {raw.height:,} rows")

    predicates_df = get_predicates_df(cfg, OmegaConf.create({"standard": "meds", "path": str(args.shard)}))
    print(f"ACES predicates_df: {predicates_df.height:,} (subject, time) rows")
    print(f"ACES trigger rows: {_summary(predicates_df.filter(pl.col(trigger) >= 1))}")

    print(f"ACES query result: {_summary(query(cfg, predicates_df))}")


if __name__ == "__main__":
    main()
