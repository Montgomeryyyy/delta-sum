import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_task_dataset",
)
def main(cfg: DictConfig) -> None:
    """
    Process MEDS and ACES shards into an NL dataset for a specific prediction task.
    Iterates directly over ACES split/shard parquet files, joins with corresponding MEDS shards,
    and aggregates all event codes per subject into a single string, filtered by prediction_time.

    Bool config argument to split on double slashes (//) for natural language formatting, or keep them for coded format.
    """
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

    
    agg_expr = (
        pl.col("code")
        .filter(pl.col("time") <= pl.col("prediction_time"))
        .sort_by("meds_row_idx")  # Relies on the row index injected below
        .str.join("\n")
    )
    if cfg.split_on_doubleslash:
        agg_expr = agg_expr.str.replace_all("//", " ", literal=True)

    for aces_path in aces_shards:
        # Extract metadata directly from the directory structure
        # Example path: aces_labels_dir/mortality_30d/train/shard_0.parquet
        task = aces_path.parent.parent.name  # "30_day_readmission"
        split = aces_path.parent.name  # "train"
        shard_filename = aces_path.name  # "shard_0.parquet"

        # Construct expected MEDS path and Output path
        meds_path = meds_dir / split / shard_filename
        out_file_path = out_dir / task / split / shard_filename

        # If the cohort has a shard, but MEDS doesn't, skip it cleanly
        if not meds_path.exists():
            log.warning(f"Missing MEDS shard for {shard_filename}. Skipping {aces_path}")
            continue
        if out_file_path.exists() and not cfg.overwrite:
            log.info(
                f"Output file {out_file_path} already exists and overwrite is {cfg.overwrite}. Skipping {aces_path}"
            )
            continue

        # Recreate the task/split directory structure in the output folder
        out_file_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Processing Task: [{task}] | Split: [{split}] | Shard: {shard_filename}")

        try:
            # 1. Lazily load dfs
            aces_lazy = pl.scan_parquet(aces_path)
            meds_lazy = pl.scan_parquet(meds_path).with_row_index("meds_row_idx")

            # 2. The Transformation Pipeline
            result = (
                aces_lazy.join(meds_lazy, on="subject_id", how="inner")
                .filter(pl.col("time") <= pl.col("prediction_time"))
                # Include boolean_value so they are retained in the output
                .group_by(["subject_id", "prediction_time", "boolean_value"])
                .agg(ehr_text=agg_expr)  # Join all codes into a single string per subject_id/prediction_time
            )

            # 3. Execute and stream to disk
            result.sink_parquet(out_file_path)
            log.debug(f"Saved processed output to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing {aces_path}: {e}")
            raise

    log.info("=== MEDS to Natural Language Dataset Processing Complete! ===")
