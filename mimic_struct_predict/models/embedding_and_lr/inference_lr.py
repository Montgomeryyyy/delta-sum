"""Inference script for linear probe (using precomputed embeddings)."""

import logging
from pathlib import Path

import hydra
import joblib
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_lr",
)
def main(cfg: DictConfig) -> None:
    """
    Load pre-computed embeddings for a single task and run logistic regression inference.
    """
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.run.dir)

    dataset_path = Path(cfg.dataset_path)
    lr_model_path = Path(cfg.lr_model_path)

    if not dataset_path.exists():
        log.error(f"Embedding file {dataset_path} not found. Exiting.")
        return

    if not lr_model_path.exists():
        log.error(f"No trained LR model found at {lr_model_path}. Exiting.")
        return

    log.info(f"Loading LR model from {lr_model_path}")
    lr_pipeline = joblib.load(lr_model_path)

    log.info(f"Loading embeddings from {dataset_path}")
    df = pl.read_parquet(dataset_path)

    if len(df) == 0:
        log.warning(f"Empty dataset at {dataset_path}. Exiting.")
        return

    dim = len(df["embedding"][0])
    # Convert to numpy array safely
    X = df["embedding"].list.to_array(dim).to_numpy()

    log.info(f"Running inference on {len(X)} examples")
    probabilities = lr_pipeline.predict_proba(X)[:, 1].tolist()

    # Save results, dropping the embedding to save space
    results_df = df.drop("embedding").with_columns(
        pl.Series("probability", probabilities),
        pl.Series("format_adherence", [1] * len(df)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "results.parquet"

    results_df.write_parquet(result_file)
    log.info(f"Saved results to {result_file}")
