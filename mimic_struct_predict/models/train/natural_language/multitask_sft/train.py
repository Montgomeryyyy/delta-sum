"""Supervised fine-tuning for multi task binary predictions."""

import logging
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
import pandas as pd
import torch
import trackio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.model_loading import load_model_and_tokenizer

log = logging.getLogger(__name__)


class TrackioTableCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        eval_dataset_dict: dict[str, datasets.Dataset],
        batch_size: int,
        random_seed: int = 42,
    ):
        self.tokenizer = tokenizer

        self.task_data = {}
        for task_name, dataset in eval_dataset_dict.items():
            # Shuffle, select the batch, and extract just the list of string prompts
            sampled_subset = dataset.shuffle(seed=random_seed).select(range(batch_size))
            self.task_data[task_name] = {
                "prompts": list(sampled_subset["prompt"]),
                "ground_truths": list(sampled_subset["completion"]),
            }

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any, **kwargs: Any) -> None:
        model.eval()
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # Ensure padding is on the left for generation

        # Prepare lists to act as our table columns
        all_prompts = []
        all_predictions = []
        all_tasks = []
        all_ground_truths = []

        # Iterate over the pre-saved static prompts, NOT the state object
        for task_name, data in self.task_data.items():
            prompts = data["prompts"]
            ground_truths = data["ground_truths"]
            # 1. Tokenize the entire batch for this specific task
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                # 2. Generate predictions for the batch
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1,
                )

            # 3. Decode the batch
            outputs = self.tokenizer.batch_decode(
                generated_ids[:, inputs["input_ids"].size(-1) :], skip_special_tokens=True
            )

            # 4. Append data to our master columns
            all_prompts.extend(prompts)
            all_predictions.extend(outputs)
            # Create a list of the task name repeated for the size of the batch
            all_tasks.extend([task_name] * len(prompts))
            all_ground_truths.extend(ground_truths)

        steps = [state.global_step] * len(all_prompts)

        df = pd.DataFrame(
            {
                "step": steps,
                "task": all_tasks,
                "prompt": all_prompts,
                "prediction": all_predictions,
                "ground_truth": all_ground_truths,
            }
        )

        trackio.log({"eval_completions": trackio.Table(dataframe=df)})

        self.tokenizer.padding_side = padding_side  # Restore original padding side


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
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    """Run SFT training.

    Args:
        cfg: Hydra configuration.
    """
    dataset_dir = Path(cfg.dataset_dir)
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_name = cfg.model.pretrained_model_name_or_path

    log.info(f"Model: {model_name}")
    log.info(f"Output directory: {run_output_dir}")

    # --- Load model and tokeniser ---
    model, tokenizer = load_model_and_tokenizer(AutoModelForCausalLM, cfg.model, eval_mode=False, padding_side="right")

    # --- Load datasets ---
    train_dataset = datasets.load_from_disk(str(dataset_dir / cfg.split_names.train))
    log.info(f"Combined train dataset has {len(train_dataset)} examples.")
    eval_dict = {
        eval_task_dir.name: datasets.load_from_disk(str(eval_task_dir))
        for eval_task_dir in (dataset_dir / cfg.split_names.val).iterdir()
    }
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
        callbacks=[TrackioTableCallback(tokenizer, eval_dict, batch_size=cfg.training.per_device_eval_batch_size)]
        if cfg.tracking.report_to == "trackio"
        else None,
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
