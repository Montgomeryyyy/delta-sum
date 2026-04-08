import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_structured_dataset",
)
def process_shards_with_labels(cfg: DictConfig):
    log.info("Starting MEDS to Structured Dataset Processing")
    
    meds_dir = Path(cfg.meds_data_dir)
    aces_dir = Path(cfg.aces_labels_dir)
    out_dir = Path(cfg.output_dir)

    aces_shards = list(aces_dir.glob("*/*/*.parquet"))
    
    if not aces_shards:
        log.error(f"No ACES parquet files found in {aces_dir}/*/*/*.parquet")
        raise ValueError("No ACES labels found. Check your directory structure.")

    log.info(f"Found {len(aces_shards)} ACES shards across tasks/splits.")

    for aces_path in aces_shards:
        task = aces_path.parent.parent.name
        split = aces_path.parent.name
        shard_filename = aces_path.name
        
        meds_path = meds_dir / split / shard_filename
        out_file_path = out_dir / task / split / shard_filename
        
        if not meds_path.exists():
            log.warning(f"Missing MEDS shard for {shard_filename}. Skipping {aces_path}")
            continue
        if out_file_path.exists() and not cfg.overwrite:
            log.info(f"Output file {out_file_path} already exists and overwrite is {cfg.overwrite}. Skipping {aces_path}")
            continue

        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Processing Task: [{task}] | Split: [{split}] | Shard: {shard_filename}")
        
        try:
            aces_lazy = pl.scan_parquet(aces_path).with_columns(
                pl.lit(task).alias("task"),
            )
            
            meds_lazy = pl.scan_parquet(meds_path)
            
            result = (
                aces_lazy
                .join(meds_lazy, on="subject_id", how="inner")
                .filter(pl.col("time") <= pl.col("prediction_time"))
                .group_by(["subject_id", "prediction_time", "boolean_value", "task"])
                .agg([
                    pl.col("code")
                    .str.join("//\n//")
                    .str.split("//")
                    .alias("code_list")
                ])
            )
            
            result.sink_parquet(out_file_path)
            log.debug(f"Saved processed output to {out_file_path}")
            
        except Exception as e:
            log.error(f"Error processing {aces_path}: {e}")
            raise

    log.info("=== MEDS to Structured Dataset Processing Complete! ===")
