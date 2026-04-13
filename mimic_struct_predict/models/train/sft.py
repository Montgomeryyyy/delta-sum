"""Supervised fine-tuning for binary predictions.

Usage:
    uv run sft_train \
        dataset_dir=data/natural_language \
        output_dir=outputs/models/sft
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Optional

import datasets
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.dataset import ehr_texts_to_prompt
from mimic_struct_predict.models.utils.model_loading import load_model_and_tokenizer

log = logging.getLogger(__name__)

DEFAULT_RANDOM_SEED = 42


class MultiTaskSFTTrainer(SFTTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        # 1. Run the standard evaluation to get the task-specific metrics
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 2. Find all keys that represent a task loss
        # This will catch 'eval_eval_icu_mortality_loss', etc.
        loss_keys = [k for k in metrics.keys() if k.endswith("_loss") and k != "eval_loss"]

        # 3. Compute the mean of those losses
        if loss_keys:
            mean_loss = sum(metrics[k] for k in loss_keys) / len(loss_keys)

            # Add it back to the metrics dictionary so load_best_model_at_end can find it
            metrics["eval_loss"] = mean_loss

            # Optional: Manually log it so it shows up in WandB/TensorBoard alongside the others
            self.log({"eval_loss": mean_loss})

        return metrics


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="sft",
)
def train_sft(cfg: DictConfig) -> None:
    """Run SFT training.

    Args:
        cfg: Hydra configuration.
    """
    dataset_dir = Path(cfg.dataset_dir)
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_name = cfg.model.pretrained_model_name_or_path
    random_seed = int(cfg.get("seed", DEFAULT_RANDOM_SEED))

    log.info(f"Model: {model_name}")
    log.info(f"Output directory: {run_output_dir}")

    # --- Load model and tokeniser ---
    model, tokenizer = load_model_and_tokenizer(AutoModelForCausalLM, cfg.model, eval_mode=False, padding_side="right")

    # --- Build datasets ---
    log.info("Building datasets...")

    # 1. Discovery shards: task/split/*.parquet
    selected_tasks_cfg = cfg.get("tasks", None)
    selected_tasks = set(selected_tasks_cfg) if selected_tasks_cfg else None
    task_to_split_to_shard_path: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for shard_path in dataset_dir.glob("*/*/*.parquet"):
        task = shard_path.parent.parent.name
        split = shard_path.parent.name
        if (
            (selected_tasks is None or task in selected_tasks)
            and task != "pretraining"
            and split in {cfg.split_names.train, cfg.split_names.val}
        ):
            task_to_split_to_shard_path[task][split].append(str(shard_path))

    if not task_to_split_to_shard_path:
        log.error(
            f"No shards found in {dataset_dir} for "
            f"splits {{{cfg.split_names.train}, {cfg.split_names.val}}} and "
            f"tasks {sorted(selected_tasks) if selected_tasks else 'ALL'}"
        )
        return
    log.info(f"Found {len(task_to_split_to_shard_path)} tasks to train on.")

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
        seed=random_seed,
        stopping_strategy="all_exhausted",  # Crucial for balancing train gradients
    )
    train_dataset = train_dataset.shuffle(seed=random_seed)
    log.info(f"Combined train dataset has {len(train_dataset)} examples.")

    eval_dict = {task: task_ds[cfg.split_names.val] for task, task_ds in task_to_datasets.items()}
    log.info(f"Eval datasets: { {task: len(ds) for task, ds in eval_dict.items()} }")

    # --- SFT config ---
    sft_config = SFTConfig(
        output_dir=str(run_output_dir),
        run_name=run_output_dir.name,
        **cfg.get("tracking", {}),
        **cfg.get("training", {}),
    )

    # --- LoRA config ---
    peft_config = None
    if "lora" in cfg:
        log.info("Using LoRA for training")
        peft_config = LoraConfig(**cfg.get("lora", {}))

    # --- Create trainer ---
    trainer = MultiTaskSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dict,
        peft_config=peft_config,
    )

    # --- Train ---
    log.info("Starting SFT training...")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", final_eval_metrics)
    trainer.save_metrics("eval", final_eval_metrics)

    # --- Save ---
    output_model_dir = run_output_dir / "final_model"
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    log.info(f"Model saved to {output_model_dir}")


if __name__ == "__main__":
    train_sft()
