import time
from pathlib import Path

import hydra
import polars as pl
import xgboost as xgb
from MEDS_tabular_automl.xgboost_model import XGBIterator, XGBoostModel
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint.

    Expected config keys (can be passed as CLI overrides):
      - trained_config_fp: path to resolved model config YAML (required)
      - model_fp: path to xgboost.json (required)
      - output_dir: directory to write predictions (required)
      - split: list of splits to predict (optional, defaults to ["held_out"]).
    """

    trained_config_fp = Path(cfg.model_fp) / "config.log"
    model_weights_fp = Path(cfg.model_fp) / "xgboost.json"
    split = cfg.split
    output_fp = Path(cfg.output_dir) / split / "results.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    # Reload model and dataset
    trained_config = OmegaConf.load(trained_config_fp)
    OmegaConf.resolve(trained_config)

    model = XGBoostModel(trained_config)
    model.load_model(model_weights_fp)

    dataset_dmatrix = xgb.DMatrix(XGBIterator(trained_config, split))

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
        },
    )

    # Save to file
    predictions_df.write_parquet(output_fp)
    print(f"Wrote {len(predictions_df)} rows to {output_fp}")
