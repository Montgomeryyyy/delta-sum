import logging
import shutil
import subprocess
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
    prediction_dir = Path(HydraConfig.get().run.dir)
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

        predictions_fp = prediction_dir / split / task_name / "results.parquet"
        predictions_fp.parent.mkdir(parents=True, exist_ok=True)

        # Reload model and dataset
        trained_config = OmegaConf.load(trained_config_fp)
        OmegaConf.resolve(trained_config)

        output_dir = Path(cfg.output_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(cfg.output_dir).parent / "metadata", output_dir / "metadata", dirs_exist_ok=True)

        # HACK: Must be a better way to do this. But allows tabularization of test set
        # using same config as the model about to be used for inference.
        tabularization_config = trained_config.tabularization

        start_time = time.perf_counter()
        command = [
            "uv",
            "run",
            "meds-tab-tabularize-time-series",
            "--multirun",
            f"worker=range(0,{cfg.num_workers})",
            "hydra/launcher=joblib",
            f"input_dir={cfg.input_dir}",
            f"output_dir={str(output_dir)}",
            f"input_label_dir={cfg.aces_dir}/{task_name}/",
            "do_overwrite=True",
            f"tabularization.window_sizes={tabularization_config.window_sizes}",
            f"tabularization.aggs={tabularization_config.aggs}",
            f"tabularization._resolved_codes={tabularization_config._resolved_codes}",
        ]

        try:
            log.info(f"Running command: {' '.join(command)[:500]}...")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            log.info(result.stdout)
        except subprocess.CalledProcessError as e:
            log.error(f"Command failed with return code {e.returncode}")
            log.error(f"Standard output: {e.stdout}")
            log.error(f"Standard error: {e.stderr}")
            raise

        trained_config.output_dir = str(output_dir)
        model = XGBoostModel(trained_config)
        model.load_model(model_weights_fp)

        dataset_dmatrix = xgb.DMatrix(XGBIterator(trained_config, split))
        log.info(
            f"Running inference for task {task_name} on split {split} with {dataset_dmatrix.num_row()} examples..."
        )
        # Run inference and time it (time only the predict call, not label assembly or IO)

        y_pred = model.model.predict(dataset_dmatrix)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Get true labels and sample latency and write df
        y_true = dataset_dmatrix.get_label()
        predictions_df = pl.DataFrame(
            {
                "boolean_value": y_true,
                "probability": y_pred,
                "format_adherence": [1] * len(y_true),
                "sample_latency": [elapsed_time / len(y_true)] * len(y_true),
            },
        )

        # Save to file
        predictions_df.write_parquet(predictions_fp)
        log.info(f"Wrote {len(predictions_df)} rows to {predictions_fp}")
