import polars as pl
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_nl_dataset",
)
def process_shards_with_labels(cfg: DictConfig):
    log.info("Starting MEDS to Natural Language Dataset Processing")
    
    meds_dir = Path(cfg.meds_data_dir)
    aces_dir = Path(cfg.aces_labels_dir)
    out_dir = Path(cfg.output_dir)

    # Find all ACES shards using rglob to match: task/split/shard_n.parquet
    aces_shards = list(aces_dir.glob("*/*/*.parquet"))
    
    if not aces_shards:
        log.error(f"No ACES parquet files found in {aces_dir}/*/*/*.parquet")
        raise ValueError("No ACES labels found. Check your directory structure.")

    log.info(f"Found {len(aces_shards)} ACES shards across tasks/splits.")

    for aces_path in aces_shards:
        # Extract metadata directly from the directory structure
        # Example path: aces_labels_dir/mortality_30d/train/shard_0.parquet
        task = aces_path.parent.parent.name  # "30_day_readmission"
        split = aces_path.parent.name        # "train"
        shard_filename = aces_path.name      # "shard_0.parquet"
        
        # Construct expected MEDS path and Output path
        meds_path = meds_dir / split / shard_filename
        out_file_path = out_dir / task / split / shard_filename
        
        # If the cohort has a shard, but MEDS doesn't, skip it cleanly
        if not meds_path.exists():
            log.warning(f"Missing MEDS shard for {shard_filename}. Skipping {aces_path}")
            continue
        if out_file_path.exists() and not cfg.overwrite:
            log.info(f"Output file {out_file_path} already exists and overwrite is {cfg.overwrite}. Skipping {aces_path}")
            continue

        # Recreate the task/split directory structure in the output folder
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Processing Task: [{task}] | Split: [{split}] | Shard: {shard_filename}")
        
        try:
            # 1. Lazily load ACES and inject the task and split columns
            aces_lazy = pl.scan_parquet(aces_path).with_columns(
                pl.lit(task).alias("task"),
            )
            
            # 2. Lazily load the matching MEDS shard
            meds_lazy = pl.scan_parquet(meds_path)
            
            # 3. The Transformation Pipeline
            result = (
                aces_lazy
                .join(meds_lazy, on="subject_id", how="inner")
                .filter(pl.col("time") <= pl.col("prediction_time"))
                # Include boolean_value and task in the group_by so they are retained in the output
                .group_by(["subject_id", "prediction_time", "boolean_value", "task"])
                .agg([
                    pl.col("code")
                    .str.join("\n")
                    .alias("code_string")
                ])
            )
            
            # 4. Execute and stream to disk
            result.sink_parquet(out_file_path)
            log.debug(f"Saved processed output to {out_file_path}")
            
        except Exception as e:
            log.error(f"Error processing {aces_path}: {e}")
            raise

    log.info("=== MEDS to Natural Language Dataset Processing Complete! ===")
