"""Supervised fine-tuning for multi task binary predictions."""

import logging
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = logging.getLogger(__name__)
torch.set_printoptions(profile="full")


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


class MultiTaskSFTTrainer(SFTTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.is_first = True

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

    def compute_loss(self, model: Any, inputs: Any, return_outputs: Any = False, num_items_in_batch: Any = None) -> Any:
        # Get loss and outputs
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # Log text at the very first step, and then every 10 steps to not spam the logs
        if self.state.is_world_process_zero and (
            self.is_first or (self.state.global_step > 0 and self.state.global_step % 10 == 0)
        ):
            self.is_first = False

            tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
            if tokenizer is not None:
                # Take the first example in the batch
                input_ids = inputs["input_ids"][0]
                labels = inputs["labels"][0]

                # Decode the input text
                input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

                # Decode the valid labels (where label != -100)
                valid_label_mask = labels != -100
                if valid_label_mask.any():
                    label_tokens = input_ids[valid_label_mask]
                    label_text = tokenizer.decode(label_tokens.cpu().tolist(), skip_special_tokens=False)
                else:
                    label_text = "None (All labels masked to -100!)"

                # Build alignment string for the last 50 tokens
                alignment_str = "Token-by-Token Alignment (Last 50 tokens):\n"
                # input_ids and labels have the same shape
                seq_len = len(input_ids)
                start_idx = max(0, seq_len - 50)
                for i in range(start_idx, seq_len):
                    # .item() safely pulls the value from the GPU to a Python int
                    in_tok = tokenizer.decode([input_ids[i].item()])
                    in_tok_clean = repr(in_tok)  # use repr to make \n and spaces visible

                    if labels[i] == -100:
                        alignment_str += f"{in_tok_clean:20} -> (masked)\n"
                    else:
                        lbl_tok = tokenizer.decode([labels[i].item()])
                        lbl_tok_clean = repr(lbl_tok)
                        alignment_str += f"{in_tok_clean:20} -> TARGET: {lbl_tok_clean}\n"

                log_str = f"\n{'-'*40} Step {self.state.global_step} {'-'*40}\n"
                log_str += f"INPUT PROMPT (Last 200 chars):\n{input_text[-200:]}\n\n"
                log_str += f"TARGET LABELS (Tokens Model is Trained On):\n{label_text}\n\n"
                log_str += f"{alignment_str}\n"
                log_str += f"{'-'*89}\n"
                log.info(log_str)

        return (loss, outputs) if return_outputs else loss


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

    log.info(f"Model: {cfg.model}")
    log.info(f"Output directory: {run_output_dir}")

    # --- Load datasets ---
    train_dataset = datasets.load_from_disk(str(dataset_dir / cfg.split_names.train))
    train_dataset = train_dataset.select_columns(["prompt", "completion"])

    log.info(
        f"Combined train dataset has {len(train_dataset)} examples and the following columns:"
        f" {train_dataset.column_names}"
    )
    eval_dict = {
        eval_task_dir.name: datasets.load_from_disk(str(eval_task_dir))
        .shuffle(seed=cfg.seed)
        # .select(range(cfg.max_eval_samples))
        .select_columns(["prompt", "completion"])
        for eval_task_dir in (dataset_dir / cfg.split_names.val).iterdir()
        if eval_task_dir.is_dir()
    }
    log.info(f"Eval datasets: { {task: (len(ds), ds.column_names) for task, ds in eval_dict.items()} }")

    # --- SFT config ---
    sft_config = SFTConfig(
        output_dir=str(run_output_dir),
        run_name=run_output_dir.name,
        **OmegaConf.to_container(cfg.get("training", {}), resolve=True),
    )

    # --- LoRA config ---
    peft_config = None
    if "lora" in cfg:
        log.info("Using LoRA for training")
        peft_config = LoraConfig(**cfg.get("lora", {}))

    # if cfg.brier_loss:
    #     log.info("Using custom Brier loss for training")
    #     digit_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
    #     true_label_token_id = tokenizer.convert_tokens_to_ids("1")
    #     custom_loss_func = create_expected_brier_loss(digit_token_ids, true_label_token_id)

    # --- Create trainer ---
    trainer = MultiTaskSFTTrainer(
        model=cfg.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dict,
        peft_config=peft_config,
        callbacks=[MultiGpuTensorBoardCallback()],
        # compute_loss_func=custom_loss_func if cfg.brier_loss else None,
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
    # tokenizer.save_pretrained(str(output_model_dir))
    log.info(f"Model saved to {output_model_dir}")
