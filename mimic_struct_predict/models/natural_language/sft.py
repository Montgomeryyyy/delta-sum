"""Supervised fine-tuning for multi task binary predictions."""

import logging
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
import pandas as pd
import torch
import torch.nn.functional as F
import trackio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.model_loading import load_model_and_tokenizer

log = logging.getLogger(__name__)


def create_expected_brier_loss(digit_token_ids: list[int], true_label_token_id: int) -> Any:
    """
    digit_token_ids: List of token IDs representing confidences 0 through N.
                     (e.g., the token IDs for '0', '1', ..., '9').
    true_label_token_id: The specific token ID in your labels that represents
                         a correctness of 1.0 (True).
    """
    N = len(digit_token_ids) - 1
    implied_confidences = torch.tensor([i / N for i in range(N + 1)], dtype=torch.float32)
    allowed_ids_tensor = torch.tensor(digit_token_ids, dtype=torch.long)

    def expected_brier_loss_func(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> torch.Tensor:
        logits = outputs.get("logits") if hasattr(outputs, "get") else outputs.logits

        device = logits.device
        confidences = implied_confidences.to(device)
        allowed_ids = allowed_ids_tensor.to(device)

        # 1. Extract final token logits & labels
        target_logits = logits[:, -2, :]  # (batch_size, vocab_size)
        target_labels = labels[:, -1]  # (batch_size,)

        # 2. Map the token ID to the mathematical y value (1.0 or 0.0)
        # If the label is the true token ID, y becomes 1.0. Otherwise 0.0.
        y = (target_labels == true_label_token_id).float()

        # 3. Extract q: the predictive distribution over the confidence tokens
        relevant_logits = target_logits[:, allowed_ids]  # (batch_size, N+1)
        q = F.softmax(relevant_logits, dim=-1)  # (batch_size, N+1)
        # 4. Compute expected squared error: sum( q_i * (y - i/N)^2 )
        squared_errors = (y.unsqueeze(1) - confidences.unsqueeze(0)) ** 2
        brier_loss_per_sample = torch.sum(q * squared_errors, dim=-1)

        # 5. Average the loss
        if num_items_in_batch is not None:
            return torch.sum(brier_loss_per_sample) / num_items_in_batch
        return torch.mean(brier_loss_per_sample)

    return expected_brier_loss_func


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
                "ground_truths": list(sampled_subset["boolean_value"]),
            }

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any, **kwargs: Any) -> None:
        model.eval()
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"  # Ensure padding is on the left for generation

        # Prepare lists to act as our table columns
        all_prompts = []
        all_completions = []
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
            all_completions.extend(outputs)
            # Create a list of the task name repeated for the size of the batch
            all_tasks.extend([task_name] * len(prompts))
            all_ground_truths.extend(ground_truths)

        steps = [state.global_step] * len(all_prompts)

        df = pd.DataFrame(
            {
                "step": steps,
                "task": all_tasks,
                "prompt": all_prompts,
                "completion": all_completions,
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
    config_name="sft",
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
    log.info(
        f"Combined train dataset has {len(train_dataset)} examples and the following columns:"
        f" {train_dataset.column_names}"
    )
    eval_dict = {
        eval_task_dir.name: datasets.load_from_disk(str(eval_task_dir))
        for eval_task_dir in (dataset_dir / cfg.split_names.val).iterdir()
        if eval_task_dir.is_dir()
    }
    log.info(f"Eval datasets: { {task: (len(ds), ds.column_names) for task, ds in eval_dict.items()} }")

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

    if cfg.brier_loss:
        log.info("Using custom Brier loss for training")
        digit_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
        true_label_token_id = tokenizer.convert_tokens_to_ids("1")
        custom_loss_func = create_expected_brier_loss(digit_token_ids, true_label_token_id)

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
        compute_loss_func=custom_loss_func if cfg.brier_loss else None,
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
