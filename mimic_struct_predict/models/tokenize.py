import logging
import re
from pathlib import Path
from typing import Iterable

import datasets
import hydra
import polars as pl
from omegaconf import DictConfig
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    PreTrainedTokenizerFast,
)

from mimic_struct_predict.models.utils.model_loading import load_tokenizer
from mimic_struct_predict.models.utils.preprocessing import (
    meds_to_string_dataset,
)

log = logging.getLogger(__name__)


def train_custom_tokenizer(dataset: datasets.Dataset, cfg: DictConfig, batch_size: int) -> PreTrainedTokenizerFast:
    # 1. Initialize the WordLevel tokenizer (no subword merging)
    tokenizer = Tokenizer(WordLevel(unk_token=cfg.special_tokens.unk_token))

    # 2. Set the Pre-Tokenizer to ONLY split on the delimited strings
    # We removed the Sequence and Whitespace() from the previous example.
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(Regex(re.escape(cfg.special_tokens.eoc_token)), behavior="isolated"),
            Split(Regex(re.escape(cfg.delimiter)), behavior="removed"),
        ]
    )

    # 3. Set up the Trainer
    trainer = WordLevelTrainer(
        min_frequency=cfg.min_frequency,
        # Assume that unk and eoc are already in text so not added as special token to allow for
        # simpler decoding
        special_tokens=[cfg.special_tokens.pad_token],
    )

    def batch_iterator() -> Iterable[list[str]]:
        # Ensure all time tokens are in vocab by yielding them min_frequency times
        time_tokens = (
            [f"+{num} days" for num in range(1, 366)]
            + [f"{num:02d}h" for num in range(24)]
            + [f"{num:02d}m" for num in range(60)]
        )
        # HACK: To ensure time_token is always in vocab
        for _ in range(max(1, cfg.min_frequency)):
            yield time_tokens

        for batch in dataset.iter(batch_size=batch_size):
            yield batch["ehr_text"]

    log.info("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=cfg.model_max_length,
        padding_side=cfg.padding_side,
        truncation_side=cfg.truncation_side,
        pad_token=cfg.special_tokens.pad_token,
    )


@hydra.main(version_base=None, config_path="config/tokenize")
def main(cfg: DictConfig) -> None:
    """
    Main function to create a pretraining dataset from MEDS, train a custom tokenizer, and tokenize the dataset.
    """

    meds_data_dir = Path(cfg.dataset_dir)
    output_dir = Path(cfg.output_dir)

    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val

    if not cfg.overwrite and (output_dir / "tokenized_dataset").exists():
        log.info(
            f"Tokenized dataset already exists at {output_dir / 'tokenized_dataset'} and overwrite is "
            f"{cfg.overwrite}. Skipping tokenization."
        )
        return

    log.info("Creating raw pretraining dataset from MEDS shards...")
    raw_data_dir = output_dir / "raw_data"
    meds_to_string_dataset(
        meds_dir=meds_data_dir,
        output_dir=raw_data_dir,
        splits={train_split, val_split},
        # Add delimiter around eoc token to ensure it is always a separate token
        # and to allow for easier decoding back to text
        overwrite=cfg.overwrite,
        **cfg.pre_processing,
    )

    log.info("Loading raw pretraining dataset as huggingface dataset...")
    datafiles: dict[str, list[str]] = {train_split: [], val_split: []}
    for shard_path in raw_data_dir.glob("*/*.parquet"):
        split = shard_path.parent.name
        if split not in {train_split, val_split}:
            continue
        datafiles[split].append(str(shard_path))
    dataset = datasets.load_dataset("parquet", data_files=datafiles)
    log.info(f"Loaded {len(dataset[train_split])} train examples and {len(dataset[val_split])} val examples.")

    if cfg.tokenizer.pre_trained_tokenizer:
        log.info(f"Loading existing tokenizer {cfg.tokenizer.pre_trained_tokenizer}")
        tokenizer = load_tokenizer(cfg.tokenizer.pre_trained_tokenizer, padding_side=cfg.tokenizer.padding_side)
    else:
        log.info("Training custom tokenizer on raw pretraining dataset...")
        tokenizer = train_custom_tokenizer(dataset[train_split], cfg.tokenizer, cfg.batch_size)
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        log.info(f"Tokenizer of vocab size {len(tokenizer)} trained and saved to {output_dir / 'tokenizer'}")

    log.info("Tokenizing and chunking raw pretraining dataset with the trained tokenizer...")
    tokenized_dataset = dataset.map(
        lambda texts: {
            k: v
            for k, v in tokenizer(
                texts,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                padding=False,
                truncation=True,
                padding_side=cfg.tokenizer.padding_side,
                max_length=cfg.tokenization.max_length,
                stride=cfg.tokenization.stride,
            ).items()
            if k in ["input_ids", "attention_mask"]
        },
        input_columns="ehr_text",
        remove_columns=dataset[train_split].column_names,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
    )

    log.info(
        f"Tokenized dataset has {len(tokenized_dataset[train_split])} train examples "
        f"and {len(tokenized_dataset[val_split])} val examples after tokenization."
    )
    # Save the tokenized dataset to disk and remove the intermediate data directory
    tokenized_dataset.save_to_disk(output_dir / "tokenized_dataset")
    log.info(f"Tokenized dataset saved to {output_dir / 'tokenized_dataset'}")

    tokenized_dataset.cleanup_cache_files()

    log.info("Counting total tokens in the tokenized training dataset...")

    def count_tokens_polars(dataset_split: datasets.Dataset) -> int:
        # Convert the HF dataset split directly to a Polars DataFrame
        df = pl.from_arrow(dataset_split.data.table)

        # Calculate the sum of the lengths of the input_ids lists
        return int(df.select(pl.col("input_ids").list.len().sum()).item())

    total_tokens = count_tokens_polars(tokenized_dataset["train"])
    log.info(f"Total tokens in the tokenized training dataset: {total_tokens}")
