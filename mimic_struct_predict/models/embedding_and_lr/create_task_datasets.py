import logging
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mimic_struct_predict.models.utils import meds_and_aces_to_string_dataset

log = logging.getLogger(__name__)


def get_tokenize_fn(tokenizer: PreTrainedTokenizerBase) -> Callable:
    """Returns a tokenization function for dataset mapping."""

    def tokenize_fn(ehr_texts: list[str]) -> Any:
        # Pad at data collator stage
        tokenized = tokenizer(ehr_texts, truncation=True, padding=False)
        tokenized["token_len"] = [len(ids) for ids in tokenized["input_ids"]]
        return tokenized

    return tokenize_fn


def tokenize_full_text_directory(
    task_to_split_to_shard_paths: dict[str, dict[str, list[Path]]], output_dir: Path, cfg: DictConfig
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(padding_side="left", truncation_side="left", **cfg.tokenizer)

    for task, split_to_shard_paths in task_to_split_to_shard_paths.items():
        dataset_dict = datasets.DatasetDict()
        for split, shard_paths in split_to_shard_paths.items():
            if not shard_paths:
                log.warning(f"No full text shards found for Task: [{task}] | Split: [{split}]")
                continue
            dataset_dict[split] = datasets.Dataset.from_parquet([str(p) for p in shard_paths])

        if not dataset_dict:
            continue

        log.info(f"Tokenizing Task: [{task}]")

        dataset_dict = dataset_dict.map(
            get_tokenize_fn(tokenizer),
            batched=True,
            num_proc=cfg.num_workers,
            desc=f"Tokenizing Task: [{task}]",
            input_columns="ehr_text",
        )

        dataset_dict = dataset_dict.sort("token_len", reverse=True)
        dataset_dict = dataset_dict.remove_columns("token_len")

        tokenized_dir = output_dir / "tokenized" / task
        tokenized_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Saving tokenized dataset to {tokenized_dir}")
        dataset_dict.save_to_disk(str(tokenized_dir))
        dataset_dict.cleanup_cache_files()

        if cfg.delete_intermediate_texts:
            log.info(f"Deleting intermediate text parquet files for {task}")
            for shard_paths in split_to_shard_paths.values():
                for shard in shard_paths:
                    shard.unlink()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_ehr_texts",
)
def main(cfg: DictConfig) -> None:
    meds_dir = Path(cfg.meds_dir)
    aces_dir = Path(cfg.aces_dir)
    output_dir = Path(cfg.output_dir)

    valid_splits = set(cfg.splits_to_process)
    valid_tasks = set(cfg.get("tasks") or [])

    log.info("Combining MEDS and ACES datasets to raw EHR text datasets...")
    full_text_output_dir = output_dir / "full_text"
    meds_and_aces_to_string_dataset(
        meds_dir=meds_dir,
        aces_dir=aces_dir,
        output_dir=full_text_output_dir,
        overwrite=cfg.overwrite,
        tasks=valid_tasks if valid_tasks else None,
        splits=valid_splits,
        **cfg.pre_processing,
    )
    log.info(f"Finished creating raw EHR text datasets. Saved to: {full_text_output_dir}")

    log.info("Proceeding with tokenization...")

    task_to_split_to_shard_paths = {
        task_dir.name: {
            split: list((task_dir / split).glob("*.parquet")) for split in valid_splits if (task_dir / split).exists()
        }
        for task_dir in full_text_output_dir.iterdir()
        if task_dir.is_dir()
    }

    tokenize_full_text_directory(
        task_to_split_to_shard_paths=task_to_split_to_shard_paths,
        output_dir=output_dir,
        cfg=cfg,
    )
