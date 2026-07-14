"""Supervised fine-tuning for multi task binary predictions."""

import logging
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from scipy.special import softmax
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, EarlyStoppingCallback, EvalPrediction, TrainerCallback
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = logging.getLogger(__name__)


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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[-1]
    no_token_id = tokenizer("No", add_special_tokens=False).input_ids[-1]

    def preprocess_logits_for_metrics(logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if isinstance(logits, tuple):
            logits = logits[0]

        # Keep only the logits for 'Yes' and 'No' tokens.
        # This changes the last dimension from vocab_size to 2
        yes_no_logits = logits[..., [yes_token_id, no_token_id]]

        # Shift so that tokens < n predict n
        shift_logits = yes_no_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Since there is exactly 1 valid label per input, we find its index using argmax.
        # This guarantees the output is exactly (batch_size, 2), even if DDP introduces
        # completely padded dummy rows (where all labels are -100).
        # Completely padded rows will just extract index 0 (which is also -100) and get safely filtered later.
        valid_indices = (shift_labels != -100).long().argmax(dim=1)
        batch_indices = torch.arange(shift_logits.shape[0], device=shift_logits.device)

        return shift_logits[batch_indices, valid_indices], shift_labels[batch_indices, valid_indices]

    def compute_metrics(eval_preds: EvalPrediction) -> dict:
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]

        # DDP gathering might pad some batches with -100 across GPUs, so we still filter -100 here.
        valid_mask = labels != -100
        valid_logits = logits[valid_mask]  # shape: (num_valid, 2)
        valid_labels = labels[valid_mask]  # shape: (num_valid,)

        if len(valid_labels) == 0:
            return {"accuracy": 0.0}

        # Calculate probabilities using softmax over the two classes
        pos_probs = softmax(valid_logits, axis=-1)[:, 0]

        # y_true is 1 if the true label is 'Yes', else 0
        y_true = (valid_labels == yes_token_id).astype(int)
        y_pred = (pos_probs >= 0.5).astype(int)

        metrics = {}
        try:
            metrics["auroc"] = roc_auc_score(y_true, pos_probs)
            metrics["ap"] = average_precision_score(y_true, pos_probs)
            metrics["f1"] = f1_score(y_true, y_pred)
        except Exception as e:
            log.warning(f"Failed to compute binary metrics (possibly due to only one class present): {e}")

        metrics["accuracy"] = (y_true == y_pred).mean()
        return metrics

    # HACK: So EOS token is not added
    original_eos = tokenizer.eos_token
    if not cfg.add_eos_token:
        tokenizer.eos_token = ""

    callbacks: list[TrainerCallback] = [MultiGpuTensorBoardCallback()]
    if getattr(cfg, "early_stopping_patience", None) is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    # --- Create trainer ---
    trainer = MultiTaskSFTTrainer(
        model=cfg.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dict,
        peft_config=peft_config,
        callbacks=callbacks,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    tokenizer.eos_token = original_eos

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
