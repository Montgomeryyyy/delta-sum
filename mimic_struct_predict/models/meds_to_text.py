"""Convert MEDS parquet shards into patient-level EHR text (`raw_data`)."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mimic_struct_predict.models.utils.preprocessing import meds_to_string_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config/meds_to_text")
def main(cfg: DictConfig) -> None:
    """Create a raw pretraining text dataset from MEDS shards."""
    meds_data_dir = Path(cfg.dataset_dir)
    output_dir = Path(cfg.output_dir)
    raw_data_dir = output_dir / "raw_data"

    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val

    if not cfg.overwrite and raw_data_dir.exists() and any(raw_data_dir.glob("*/*.parquet")):
        log.info(
            f"Raw pretraining dataset already exists at {raw_data_dir} and overwrite is "
            f"{cfg.overwrite}. Skipping MEDS -> text conversion."
        )
        return

    log.info("Creating raw pretraining dataset from MEDS shards...")
    meds_to_string_dataset(
        meds_dir=meds_data_dir,
        output_dir=raw_data_dir,
        splits={train_split, val_split},
        overwrite=cfg.overwrite,
        **cfg.pre_processing,
    )
    log.info(f"Saved raw pretraining dataset to {raw_data_dir}")


if __name__ == "__main__":
    main()
