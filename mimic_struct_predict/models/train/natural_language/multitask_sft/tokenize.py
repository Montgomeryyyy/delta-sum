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
    config_name="tokenize",
)
def main(cfg: DictConfig) -> None:
    meds_dir = Path(cfg.meds_dir)
    aces_dir = Path(cfg.aces_dir)
    output_dir = Path(cfg.output_dir)

    if not cfg.overwrite and (
        (output_dir / cfg.split_names.train).exists() or (output_dir / cfg.split_names.val).exists()
    ):
        log.info(f"Dataset already exists at {output_dir} and overwrite is " f"{cfg.overwrite}. Skipping tokenization.")
        return

    tokenizer = load_tokenizer(cfg.model.pretrained_model_name_or_path, padding_side="right")

    # --- Build datasets ---
    log.info("Combining MEDS and ACES datasets to raw EHR text datasets...")
    raw_output_dir = output_dir / "raw"
    selected_tasks = set(cfg.tasks) if cfg.get("tasks", None) else None
    meds_and_aces_to_string_dataset(
        meds_dir=meds_dir,
        aces_dir=aces_dir,
        output_dir=raw_output_dir,
        overwrite=cfg.overwrite,
        tasks=selected_tasks,
        splits={cfg.split_names.train, cfg.split_names.val},
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
        if split in {cfg.split_names.train, cfg.split_names.val}:
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

        if cfg.split_names.train not in split_to_shards or cfg.split_names.val not in split_to_shards:
            log.warning(
                f"Skipping task '{task}' because it is missing one of required splits "
                f"[{cfg.split_names.train}, {cfg.split_names.val}]"
            )
            continue

        dataset = datasets.load_dataset("parquet", data_files=split_to_shards, columns=["ehr_text", "boolean_value"])
        log.info(
            f"Loaded {len(dataset[cfg.split_names.train])} train examples and "
            f"{len(dataset[cfg.split_names.val])} val examples."
        )

        dataset = dataset.map(
            lambda ehr_texts: ehr_texts_to_prompt(
                ehr_texts=ehr_texts,
                task_description=task_description,
                tokenizer=tokenizer,
                max_prompt_length=cfg.tokenization.max_prompt_length,
                enable_thinking=False,  # Not training a thinking model so not required
                max_new_tokens=None,
            ),
            input_columns=["ehr_text"],
            remove_columns=["ehr_text"],
            batched=True,
            batch_size=cfg.tokenization.batch_size,
        )

        dataset = dataset.map(
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

    if not task_to_datasets:
        log.error("No tasks had both required splits after filtering; exiting training.")
        return

    train_dataset = datasets.interleave_datasets(
        [task_dataset[cfg.split_names.train] for task_dataset in task_to_datasets.values()],
        probabilities=[1.0 / len(task_to_datasets)] * len(task_to_datasets),
        seed=cfg.random_seed,
        stopping_strategy="all_exhausted",  # Crucial for balancing train gradients
    )
    train_dataset = train_dataset.shuffle(seed=cfg.random_seed)
    log.info(f"Combined train dataset has {len(train_dataset)} examples.")

    eval_dict = {task: task_ds[cfg.split_names.val] for task, task_ds in task_to_datasets.items()}
    log.info(f"Eval datasets: { {task: len(ds) for task, ds in eval_dict.items()} }")

    # --- Save tokenized datasets ---
    tokenized_output_dir = output_dir / "tokenized_dataset"
    tokenized_output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.save_to_disk(str(tokenized_output_dir / cfg.split_names.train))
    for task, eval_ds in eval_dict.items():
        eval_ds.save_to_disk(str(tokenized_output_dir / cfg.split_names.val / task))
    log.info(f"Saved tokenized datasets to {tokenized_output_dir}")
