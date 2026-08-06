"""Inference script for structured generative rollout."""

import logging
import math
import shutil
from pathlib import Path
from typing import Any

import hydra
import joblib
import polars as pl
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train",
)
def main(cfg: DictConfig) -> float:
    """
    For a task:
    1. Load the generated embeddings dataset for the training and validation splits.
    2. Train a logistic regression model on the training embeddings
    3. evaluate on the validation set, logging the AUC.
    4. Save the trained logistic regression model to disk.
    """

    task_dir = Path(cfg.task_dir)
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir if "sweep" in hydra_cfg else Path(hydra_cfg.run.dir)

    task_name = task_dir.name

    # Load datasets
    log.info(f"Loading data from task directory {task_dir}")
    train_embedding_path = task_dir / cfg.split_names.train / "embeddings.parquet"
    val_embedding_path = task_dir / cfg.split_names.val / "embeddings.parquet"

    train_df = pl.read_parquet(train_embedding_path, columns=["embedding", "boolean_value"])
    val_df = pl.read_parquet(val_embedding_path, columns=["embedding", "boolean_value"])
    # Convert list of embeddings to 2D array for sklearn, and labels to 1D array
    dim = len(train_df["embedding"][0])
    train_X, train_y = (
        train_df["embedding"].list.to_array(dim).to_numpy(),
        train_df["boolean_value"].cast(pl.Int8).to_numpy(),
    )
    val_X, val_y = val_df["embedding"].list.to_array(dim).to_numpy(), val_df["boolean_value"].cast(pl.Int8).to_numpy()

    log.info(
        f"Loaded embeddings dataset for task {task_name} with {len(train_X)} training examples and "
        f"{len(val_X)} validation examples."
    )

    pos_percent_train = (train_y.sum() / len(train_y)) * 100
    pos_percent_val = (val_y.sum() / len(val_y)) * 100

    if train_y.sum() == 0 or val_y.sum() == 0:
        log.warning(
            f"Warning: One of the splits for task {task_name} has 0 positive examples. This will cause AUC to be "
            f"undefined and likely lead to errors during training. Percentage of positive labels in training set: "
            f"{pos_percent_train:.2f}%. Percentage of positive labels in validation set: {pos_percent_val:.2f}%."
        )
        return 0.0

    log.info(f"Percentage of positive labels in training set: {train_y.mean() * 100:.2f}%")
    log.info(f"Percentage of positive labels in validation set: {val_y.mean() * 100:.2f}%")

    # Set up and train logistic regression model
    clf = make_pipeline(
        Normalizer(norm="l2"),
        LogisticRegression(
            C=cfg.lr.C,
            solver=cfg.lr.solver,
            max_iter=cfg.lr.max_iter,
            random_state=cfg.random_seed,
            # If the sweep chose "none", pass None to LogisticRegression
            class_weight=cfg.lr.class_weight if cfg.lr.class_weight != "none" else None,
        ),
    )
    train_X, train_y = shuffle(train_X, train_y, random_state=cfg.random_seed)
    clf.fit(train_X, train_y)

    lr_step = clf.named_steps["logisticregression"]
    actual_iterations = lr_step.n_iter_[0]
    if actual_iterations == lr_step.max_iter:
        log.warning(f"Trial failed to converge! Hit max_iter of {lr_step.max_iter}")

    # Evaluate on validation set
    auc = float(roc_auc_score(val_y, clf.predict_proba(val_X)[:, 1]))
    log.info(f"Validation AUC for task {task_name}: {auc:.4f}")

    # Save the trained model to disk
    model_path = out_dir / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    log.info(f"Model successfully saved to {model_path}")

    return auc


class CleanupCallback(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """Callback to clean up suboptimal trials after a Optuna sweep finishes.

        Think there must be a more elegant way to do this, but this is a start. It:
        1. Reads the optimization_results.yaml file to identify the best trial's parameters.
        2. Iterates through the trial directories to find the one matching the best parameters.
        3. Renames the best trial's directory to "best_trial" for easy access
        4. Deletes all other trial directories to save space.
        """

        sweep_dir = Path(config.hydra.sweep.dir)
        results_file = sweep_dir / "optimization_results.yaml"

        # 1. Get best_params
        if not results_file.exists():
            log.warning(f"Could not find {results_file}. Aborting cleanup.")
            return
        opt_results = yaml.safe_load(results_file.read_text())
        best_params = opt_results.get("best_params")
        if not best_params:
            log.warning("No 'best_params' found in optimization_results.yaml. Aborting.")
            return

        best_trial_num = None
        for folder in sweep_dir.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                trial_cfg_path = folder / ".hydra" / "config.yaml"

                if trial_cfg_path.exists():
                    trial_cfg = OmegaConf.load(trial_cfg_path)

                    is_match = True
                    for param_key, param_val in best_params.items():
                        trial_val = OmegaConf.select(trial_cfg, param_key)
                        if isinstance(param_val, float) and isinstance(trial_val, float):
                            if not math.isclose(param_val, trial_val, rel_tol=1e-5):
                                is_match = False
                                break
                        # Handle everything else
                        elif trial_val != param_val:
                            is_match = False
                            break

                    if is_match:
                        best_trial_num = folder.name
                        break

        if best_trial_num is None:
            log.warning("Could not find a trial matching the best parameters. Aborting cleanup.")
            return

        best_value = opt_results.get("best_value", "Unknown")
        log.info(f"Sweep Finished. Best trial identified as: Trial {best_trial_num} (Value: {best_value})")

        # 2. Rename best model to the root of the sweep directory
        winning_trial_dir = sweep_dir / str(best_trial_num)
        best_trial_dir = sweep_dir / "best_trial"
        if best_trial_dir.exists():
            shutil.rmtree(best_trial_dir)
        log.info(f"Renaming best model from {winning_trial_dir} to {sweep_dir / 'best_trial'}")
        winning_trial_dir.rename(sweep_dir / "best_trial")

        # 4. Delete all other trial directories
        log.info(f"Cleaning sub optimal trial directories in {sweep_dir}...")
        for folder in sweep_dir.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                shutil.rmtree(folder)

        log.info("Sweep cleanup complete.")
