import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mimic_struct_predict.models.utils.meds_to_string_dataset import meds_and_aces_to_string_dataset

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_task_prompts",
)
def main(cfg: DictConfig) -> None:
    meds_dir = Path(cfg.meds_dir)
    aces_dir = Path(cfg.aces_dir)
    output_dir = Path(cfg.output_dir)

    valid_splits = set(cfg.splits_to_process)
    valid_tasks = set(cfg.get("tasks") or [])

    # --- Build datasets ---
    log.info("Combining MEDS and ACES datasets to raw EHR text datasets per task...")
    meds_and_aces_to_string_dataset(
        meds_dir=meds_dir,
        aces_dir=aces_dir,
        output_dir=output_dir,
        overwrite=cfg.overwrite,
        tasks=valid_tasks if valid_tasks else None,
        splits=valid_splits,
        code_join_token=cfg.preprocessing.code_join_token,
        split_on_delimiter=False,  # Tokenizer will split to create tokens
        append_numeric_values=False,  # We are using quantiles instead of raw numeric values
    )
