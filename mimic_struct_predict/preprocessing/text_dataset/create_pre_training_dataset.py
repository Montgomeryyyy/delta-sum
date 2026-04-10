import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_pre_training_dataset",
)
def main(cfg: DictConfig) -> None:
    """Process MEDS shards into an NL dataset without ACES label filtering.

    Iterates directly over MEDS split/shard parquet files and aggregates all
    event codes per subject into a single string, with no prediction_time filter.

    Bool config argument to split on double slashes (//) for natural language formatting, or keep them for coded format.
    """
    log.info("Starting MEDS to Natural Language Dataset Processing (Pre training)")

    meds_dir = Path(cfg.meds_data_dir)
    out_dir = Path(cfg.output_dir)

    # Find all MEDS shards: split/shard.parquet
    meds_shards = list(meds_dir.glob("*/*.parquet"))

    if not meds_shards:
        log.error(f"No MEDS parquet files found under {meds_dir}/*/*.parquet")
        raise ValueError("No MEDS shards found. Check your directory structure.")

    log.info(f"Found {len(meds_shards)} MEDS shards.")

    agg_expr = pl.col("code").str.join("\n")
    if cfg.split_on_doubleslash:
        agg_expr = agg_expr.str.replace_all("//", " ", literal=True)

    for meds_path in meds_shards:
        split = meds_path.parent.name  # e.g. "train"
        shard_filename = meds_path.name  # e.g. "0.parquet"

        out_file_path = out_dir / "pre_training" / split / shard_filename

        if out_file_path.exists() and not cfg.overwrite:
            log.info(f"Output file {out_file_path} already exists and overwrite is " f"{cfg.overwrite}. Skipping.")
            continue

        out_file_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Processing Split: [{split}] | Shard: {shard_filename}")

        try:
            result = pl.scan_parquet(meds_path).group_by("subject_id").agg(agg_expr.alias("ehr_text"))

            result.sink_parquet(out_file_path)
            log.info(f"Saved processed output to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing {meds_path}: {e}")
            raise

    log.info("=== MEDS to Natural Language Dataset Processing (Pre training) Complete! ===")
