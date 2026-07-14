"""Supervised fine-tuning for multi task binary predictions."""

import logging
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = logging.getLogger(__name__)


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
    log.info(f"Model saved to {output_model_dir}")

    if peft_config is not None:
        # Wait for all GPUs to finish saving the adapter
        trainer.accelerator.wait_for_everyone()

        if trainer.is_world_process_zero():
            log.info("Merging LoRA weights with base model...")
            merged_model_dir = run_output_dir / "merged_model"

            # Extract the actual dtype used during training before deleting the trainer
            hf_dtype = getattr(trainer.model, "dtype", torch.float16)

            # Free memory before merging
            del trainer
            torch.cuda.empty_cache()

            log.info(f"Loading model for merging with dtype: {hf_dtype}")
            model = AutoPeftModelForCausalLM.from_pretrained(
                str(output_model_dir),
                device_map="auto",
                torch_dtype=hf_dtype,
            )
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged_model_dir))

            tokenizer = AutoTokenizer.from_pretrained(str(output_model_dir))
            tokenizer.save_pretrained(str(merged_model_dir))

            log.info(f"Merged model saved to {merged_model_dir}")
