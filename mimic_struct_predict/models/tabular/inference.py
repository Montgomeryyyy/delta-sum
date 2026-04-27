import logging
import time
from pathlib import Path

import hydra
import polars as pl
import xgboost as xgb
from hydra.core.hydra_config import HydraConfig
from MEDS_tabular_automl.xgboost_model import XGBIterator, XGBoostModel
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Expected config keys (can be passed as CLI overrides):
      - model_dir: directory containing trained models (required)
      - output_dir: directory to write predictions (required)
      - split: list of splits to predict (optional, defaults to ["held_out"]).

    For each task in model_dir,
        - finds the latest trial,
        - loads the model and config
        - runs inference on the specified split
        - writes predictions to output_dir/task/split/results.parquet.
    """

    split = cfg.split
    output_dir = Path(HydraConfig.get().run.dir)
    model_dir = Path(cfg.model_dir)

    for task_dir in model_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        # Get latest trial
        trial_dirs = sorted(task_dir.glob("*/best_trial"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not trial_dirs:
            log.warning(f"No trial dirs found for task {task_name} in {task_dir}")
            continue
        best_trial_dir = trial_dirs[0]
        log.info(f"Predicting for task {task_name} using latest model from {best_trial_dir}")

        # Sort out filepaths
        trained_config_fp = best_trial_dir / "config.log"
        model_weights_fp = best_trial_dir / "xgboost.json"

        output_fp = output_dir / split / task_name / "results.parquet"
        output_fp.parent.mkdir(parents=True, exist_ok=True)

        # Reload model and dataset
        trained_config = OmegaConf.load(trained_config_fp)
        OmegaConf.resolve(trained_config)

        model = XGBoostModel(trained_config)
        model.load_model(model_weights_fp)

        dataset_dmatrix = xgb.DMatrix(XGBIterator(trained_config, split))
        log.info(
            f"Running inference for task {task_name} on split {split} with {dataset_dmatrix.num_row()} examples..."
        )
        # Run inference and time it (time only the predict call, not label assembly or IO)

        start = time.perf_counter()
        y_pred = model.model.predict(dataset_dmatrix)
        end = time.perf_counter()

        # Get true labels and sample latency and write df
        y_true = dataset_dmatrix.get_label()
        sample_latency = (end - start) / len(y_true)
        predictions_df = pl.DataFrame(
            {
                "boolean_value": y_true,
                "probability": y_pred,
                "sample_latency": [sample_latency] * len(y_true),
                "format_adherence": [1] * len(y_true),
            },
        )

        # Save to file
        predictions_df.write_parquet(output_fp)
        log.info(f"Wrote {len(predictions_df)} rows to {output_fp}")
