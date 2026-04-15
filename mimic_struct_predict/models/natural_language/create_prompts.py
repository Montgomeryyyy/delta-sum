import logging
from collections import defaultdict
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig

from mimic_struct_predict.models.utils.dataset import ehr_texts_to_prompt
from mimic_struct_predict.models.utils.meds_to_string_dataset import meds_and_aces_to_string_dataset
from mimic_struct_predict.models.utils.model_loading import load_tokenizer

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_prompts_direct",
)
def main(cfg: DictConfig) -> None:
    meds_dir = Path(cfg.meds_dir)
    aces_dir = Path(cfg.aces_dir)
    output_dir = Path(cfg.output_dir)

    tokenizer = load_tokenizer(cfg.model.pretrained_model_name_or_path, padding_side="right")

    valid_splits = set(cfg.splits_to_process)
    valid_tasks = set(cfg.get("tasks") or [])

    # --- Build datasets ---
    log.info("Combining MEDS and ACES datasets to raw EHR text datasets...")
    raw_output_dir = output_dir / "raw"
    meds_and_aces_to_string_dataset(
        meds_dir=meds_dir,
        aces_dir=aces_dir,
        output_dir=raw_output_dir,
        overwrite=cfg.overwrite,
        tasks=valid_tasks if valid_tasks else None,
        splits=valid_splits,
        code_join_token=cfg.preprocessing.code_join_token,
        split_on_delimiter=True,
        delimiter=cfg.preprocessing.delimiter,
        append_numeric_values=True,
        sig_figs=cfg.preprocessing.numeric_value_sig_figs,
    )

    task_to_split_to_shard_path: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for shard_path in raw_output_dir.glob("*/*/*.parquet"):
        task = shard_path.parent.parent.name
        split = shard_path.parent.name

        if split in valid_splits and (not valid_tasks or task in valid_tasks):
            task_to_split_to_shard_path[task][split].append(str(shard_path))

    log.info(f"Created {len(task_to_split_to_shard_path)} raw tasks datasets.")

    task_to_datasets = {}
    for task, split_to_shards in task_to_split_to_shard_path.items():
        log.info(f"Processing Task: [{task}]")

        try:
            task_description = cfg.task_description[task]
        except KeyError:
            log.error(f"Task description not found for task: {task}")
            raise ValueError(f"Task description not found for task: {task}")

        dataset = datasets.load_dataset("parquet", data_files=split_to_shards, columns=["ehr_text", "boolean_value"])

        # Safely handle and remove empty splits so they don't persist to tokenization/interleaving
        empty_splits = []
        for split, split_dataset in dataset.items():
            if len(split_dataset) == 0:
                log.warning(f"No examples found for task {task} and split {split}. Marking for removal.")
                empty_splits.append(split)
            else:
                log.info(f"Loaded {len(split_dataset)} {split} examples")

        for split in empty_splits:
            del dataset[split]

        if not dataset:
            log.warning(f"All splits empty for task {task}. Skipping task entirely.")
            continue
        dataset = dataset.map(
            lambda ehr_texts: ehr_texts_to_prompt(
                ehr_texts=ehr_texts,
                task_description=task_description,
                tokenizer=tokenizer,
                max_prompt_length=cfg.tokenization.max_prompt_length,
                enable_thinking=cfg.tokenization.enable_thinking,
                max_new_tokens=cfg.tokenization.max_new_tokens,
            ),
            input_columns=["ehr_text"],
            remove_columns=["ehr_text"],
            batched=True,
            batch_size=cfg.tokenization.batch_size,
        )

        if cfg.train_split_name in dataset:
            # Train dataset needs to have "Yes"/"No" completions for supervised fine-tuning instead of boolean labels
            dataset[cfg.train_split_name] = dataset[cfg.train_split_name].map(
                lambda boolean_values: {
                    "completion": [
                        cfg.tokenization.yes_token if boolean_value else cfg.tokenization.no_token
                        for boolean_value in boolean_values
                    ]
                },
                batched=True,
                batch_size=cfg.tokenization.batch_size,
                input_columns=["boolean_value"],
                remove_columns=["boolean_value"],
            )

        task_to_datasets[task] = dataset

    hf_output_dir = output_dir / "hf_dataset"
    hf_output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Final datasets: {task_to_datasets}")

    for split in valid_splits:
        if split == cfg.train_split_name:
            # Only collect tasks that actually have a valid, populated train split
            train_datasets = [ds[split] for ds in task_to_datasets.values() if split in ds and len(ds[split]) > 0]

            if not train_datasets:
                log.warning("No valid train datasets found across any tasks!")
                continue

            # Interleave train datasets from all tasks to ensure balanced training and shuffle to ensure
            # random mixing of tasks
            train_dataset = datasets.interleave_datasets(
                train_datasets,
                probabilities=[1.0 / len(train_datasets)] * len(train_datasets),
                seed=cfg.random_seed,
                stopping_strategy="all_exhausted",  # Crucial for balancing train gradients
            )
            train_dataset = train_dataset.shuffle(seed=cfg.random_seed)
            train_dataset.save_to_disk(str(hf_output_dir / split))
            log.info(f"Interleaved train dataset: {len(train_dataset)} examples saved to {hf_output_dir / split}")

        else:
            # Keep val/test datasets separate by task to allow for per-task evaluation
            log.info(f"Saving {split} datasets for each task separately...")
            for task, task_ds in task_to_datasets.items():
                if split in task_ds and len(task_ds[split]) > 0:
                    task_ds[split].save_to_disk(str(hf_output_dir / split / task))
                    log.info(f"{task} dataset: {len(task_ds[split])} examples saved to {hf_output_dir / split / task}")
