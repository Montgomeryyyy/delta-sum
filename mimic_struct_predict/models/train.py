import math
import os
from pathlib import Path

import datasets
import hydra
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="config/train")
def main(cfg: DictConfig) -> None:
    Accelerator()

    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(cfg.tokenizer_dir)
    dataset_dir = Path(cfg.dataset_dir)
    if not dataset_dir.exists():
        log.info(f"Tokenized dataset not found at {dataset_dir}. Please run the tokenizer script first.")
        raise FileNotFoundError(f"Tokenized dataset not found at {dataset_dir}")

    log.info("Loading dataset...")
    dataset = datasets.load_from_disk(str(dataset_dir))
    if len(dataset[cfg.split_names.val]) > cfg.max_eval_samples:
        dataset[cfg.split_names.val] = (
            dataset[cfg.split_names.val].shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
        )
    log.info(
        f"Dataset loaded. Train size: {len(dataset[cfg.split_names.train])},"
        f"Val size: {len(dataset[cfg.split_names.val])}"
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    hydra_model_config = OmegaConf.to_container(getattr(cfg.model, "config", {}), resolve=True)

    if cfg.model.random_init:
        log.info("Initializing model with random weights...")
        config = AutoConfig.from_pretrained(
            cfg.model.name,
            vocab_size=math.ceil(len(tokenizer) / cfg.model.vocab_padding_multiple) * cfg.model.vocab_padding_multiple,
            n_positions=tokenizer.model_max_length,
            pad_token_id=tokenizer.pad_token_id,
            **hydra_model_config,
        )
        model = AutoModelForCausalLM.from_config(config)
    else:
        log.info(f"Loading pre-trained model from {cfg.model.name}...")

        model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **hydra_model_config)
    model.train()
    log.info(f"Model dtype: {model.dtype}")

    if cfg.get("lora", None):
        log.info("Applying LoRA adapters to the model...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **cfg.lora,
        )
        model = get_peft_model(model, lora_config)

        # Print trainable parameters to verify LoRA is working
        trainable_params, all_param = model.get_nb_trainable_parameters()
        log.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    log.info("Starting training...")
    log.info(f"Saving outputs to {run_output_dir}")

    hydra_training_args = OmegaConf.to_container(getattr(cfg, "training_args", {}), resolve=True)
    training_args = TrainingArguments(
        output_dir=str(run_output_dir), run_name=run_output_dir.name, **hydra_training_args
    )
    if training_args.report_to == "tensorboard":
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_output_dir)
    log.info(training_args)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=cfg.pad_to_multiple_of)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset[cfg.split_names.train],
        eval_dataset=dataset[cfg.split_names.val],
        callbacks=[MultiGpuTensorBoardCallback()],
    )

    # --- Train ---
    train_result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))
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
