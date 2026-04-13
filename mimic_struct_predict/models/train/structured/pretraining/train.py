import math
from logging import getLogger
from pathlib import Path
from typing import Any

import datasets
import hydra
import trackio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

log = getLogger(__name__)


class TrackioSystemMetricsCallback(TrainerCallback):
    """Custom callback to log GPU and system metrics to Trackio."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Automatically detects NVIDIA or Apple Silicon GPUs and logs stats
        trackio.log_gpu()


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    hf_logging.set_verbosity_info()
    hf_logging.enable_propagation()

    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(cfg.tokenizer_dir)
    dataset_dir = tokenizer_dir / "tokenized_dataset"
    if not dataset_dir.exists():
        log.info(f"Tokenized dataset not found at {dataset_dir}. Please run the tokenizer script first.")
        raise FileNotFoundError(f"Tokenized dataset not found at {dataset_dir}")

    log.info("Loading dataset...")
    dataset = datasets.load_from_disk(str(dataset_dir))
    log.info(
        f"Dataset loaded. Train size: {len(dataset[cfg.split_names.train])},"
        f"Val size: {len(dataset[cfg.split_names.val])}"
    )

    log.info("Initializing model...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    config = AutoConfig.from_pretrained(
        cfg.model.name,
        vocab_size=math.ceil(len(tokenizer) / cfg.model.vocab_padding_multiple) * cfg.model.vocab_padding_multiple,
        n_positions=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.train()
    model_size = sum(t.numel() for t in model.parameters())
    log.info(f"Model size: {model_size/1000**2:.1f}M parameters")

    log.info("Starting training...")
    args = TrainingArguments(
        output_dir=str(run_output_dir), run_name=run_output_dir.name, **cfg.get("training_args", {})
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=cfg.pad_to_multiple_of)
    callbacks = [TrackioSystemMetricsCallback()] if cfg.training_args.report_to == "trackio" else None
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset[cfg.split_names.train],
        eval_dataset=dataset[cfg.split_names.val],
        callbacks=callbacks,
    )

    # --- Train ---
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
