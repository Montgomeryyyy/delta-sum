import re
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable

import datasets
import hydra
import trackio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import WordLevelTrainer
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

log = getLogger(__name__)


def load_pre_training_dataset(
    dataset_dir: Path, train_split: str, val_split: str, text_column: str
) -> datasets.DatasetDict:
    # 1. Discovery shards: split/*.parquet
    datafiles: dict[str, list[str]] = {train_split: [], val_split: []}
    for shard_path in dataset_dir.glob("*/*.parquet"):
        split = shard_path.parent.name
        if split not in {train_split, val_split}:
            continue
        datafiles[split].append(str(shard_path))

    if not datafiles[train_split] or not datafiles[val_split]:
        log.error(f"No shards found in {dataset_dir} for either split '{train_split}' or '{val_split}'. ")
        raise ValueError(f"No shards found for splits '{train_split}' or '{val_split}' in {dataset_dir}.")
    log.info(f"Found {len(datafiles[train_split])} train shards and {len(datafiles[val_split])} val shards.")

    dataset = datasets.load_dataset("parquet", data_files=datafiles, columns=[text_column])
    log.info(f"Loaded {len(dataset[train_split])} train examples and " f"{len(dataset[val_split])} val examples.")

    return dataset


def train_custom_tokenizer(dataset: datasets.Dataset, cfg: DictConfig, text_column: str) -> PreTrainedTokenizerFast:
    # 1. Initialize the WordLevel tokenizer (no subword merging)
    tokenizer = Tokenizer(WordLevel(unk_token=cfg.special_tokens.unk_token))

    # 2. Set the Pre-Tokenizer to ONLY split on the delimited strings
    # We removed the Sequence and Whitespace() from the previous example.
    tokenizer.pre_tokenizer = Sequence(
        [
            # Split on delimiter (//) and remove them
            Split(Regex(re.escape(cfg.delimiter)), behavior="removed"),
            # Split on end of code token (\n) and keep it as its own token
            Split(Regex(re.escape(cfg.special_tokens.eoc_token)), behavior="isolated"),
        ]
    )

    # 3. Set up the Trainer
    trainer = WordLevelTrainer(
        # Unk and eoc tokens not passed as are in training data and so adding here causes "holes" in the vocab.
        special_tokens=[cfg.special_tokens.pad_token],
        min_frequency=cfg.min_frequency,
    )

    def batch_iterator() -> Iterable[str]:
        for batch in dataset.iter(batch_size=cfg.training_batch_size):
            yield batch[text_column]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=cfg.model_max_length,
        padding_side=cfg.padding_side,
        truncation_side=cfg.truncation_side,
        pad_token=cfg.special_tokens.pad_token,
        unk_token=cfg.special_tokens.unk_token,
    )


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


@hydra.main(version_base=None, config_path="config", config_name="pre_training_coded")
def main(cfg: DictConfig) -> None:
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val

    log.info("Loading pre-training dataset...")
    dataset = load_pre_training_dataset(
        dataset_dir=Path(cfg.dataset_dir),
        train_split=train_split,
        val_split=val_split,
        text_column=cfg.dataset.text_column,
    )

    log.info("Training custom tokenizer on pre-training dataset...")
    tokenizer = train_custom_tokenizer(dataset["train"], cfg.tokenization, cfg.dataset.text_column)
    tokenizer.save_pretrained(str(run_output_dir / "tokenizer"))
    log.info(f"Tokenizer of size {len(tokenizer)} trained and saved to {run_output_dir / 'tokenizer'}")

    log.info("Tokenizing and chunking pre-training dataset with the trained tokenizer...")
    tokenized_dataset = dataset.map(
        lambda texts: {
            k: v
            for k, v in tokenizer(
                texts,
                truncation=True,
                stride=cfg.tokenization.stride,
                return_overflowing_tokens=True,
                padding_side="right",
                return_attention_mask=True,
                padding=False,
            ).items()
            if k in ["input_ids", "attention_mask"]
        },
        input_columns=[cfg.dataset.text_column],
        remove_columns=[cfg.dataset.text_column],
        batched=True,
    )
    log.info(
        f"Tokenized dataset has {len(tokenized_dataset[train_split])} train examples "
        f"and {len(tokenized_dataset[val_split])} val examples after tokenization."
    )

    log.info("Initializing model...")

    config = AutoConfig.from_pretrained(
        cfg.model.name,
        vocab_size=len(tokenizer),
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
    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=cfg.tokenization.pad_to_multiple_of
    )
    callbacks = [TrackioSystemMetricsCallback()] if cfg.training_args.report_to == "trackio" else None
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset[train_split],
        eval_dataset=tokenized_dataset[val_split],
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
