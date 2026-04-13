import logging
import re
import shutil
from pathlib import Path
from typing import Iterable

import datasets
import hydra
import polars as pl
from omegaconf import DictConfig
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    PreTrainedTokenizerFast,
)

log = logging.getLogger(__name__)

TEXT_COLUMN = "ehr_text"


def create_pretraining_dataset(
    meds_dir: Path, output_dir: Path, splits: set[str], eoc_token: str, overwrite: bool
) -> None:
    """Process MEDS shards into an NL dataset without ACES label filtering.

    Iterates directly over MEDS split/shard parquet files and aggregates all
    event codes per subject into a single string, with no prediction_time filter.

    Bool config argument to split on double slashes (//) for natural language formatting, or keep them for coded format.
    """
    # Find all MEDS shards: split/shard.parquet
    meds_shards = sorted(shard for shard in meds_dir.glob("*/*.parquet") if shard.parent.name in splits)

    if not meds_shards:
        log.error(f"No MEDS parquet files found under {meds_dir}/*/*.parquet")
        raise ValueError("No MEDS shards found. Check your directory structure.")

    log.info(f"Found {len(meds_shards)} MEDS shards.")

    for meds_path in meds_shards:
        split = meds_path.parent.name  # e.g. "train"
        shard_filename = meds_path.name  # e.g. "0.parquet"

        out_file_path = output_dir / split / shard_filename

        if out_file_path.exists() and not overwrite:
            log.info(f"Output file {out_file_path} already exists and overwrite is " f"{overwrite}. Skipping.")
            continue

        out_file_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Processing Split: [{split}] | Shard: {shard_filename}")

        try:
            result = (
                pl.scan_parquet(meds_path)
                .group_by("subject_id")
                .agg(pl.col("code").str.join(eoc_token).alias(TEXT_COLUMN))
                .select(TEXT_COLUMN)
            )

            result.sink_parquet(out_file_path)
            log.info(f"Saved processed output to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing {meds_path}: {e}")
            raise


def load_pretraining_dataset(dataset_dir: Path, train_split: str, val_split: str) -> datasets.DatasetDict:
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

    dataset = datasets.load_dataset("parquet", data_files=datafiles, columns=[TEXT_COLUMN])

    return dataset


def train_custom_tokenizer(dataset: datasets.Dataset, cfg: DictConfig, batch_size: int) -> PreTrainedTokenizerFast:
    # 1. Initialize the WordLevel tokenizer (no subword merging)
    tokenizer = Tokenizer(WordLevel(unk_token=cfg.special_tokens.unk_token))
    # Add special tokens directly to tokenizer rather than in trainer to avoid "Vocab holes"
    # when special tokens occur in training data.
    # Also ensures eoc_token is an individual token
    tokenizer.add_special_tokens(
        [cfg.special_tokens.pad_token, cfg.special_tokens.unk_token, cfg.special_tokens.eoc_token]
    )

    # 2. Set the Pre-Tokenizer to ONLY split on the delimited strings
    # We removed the Sequence and Whitespace() from the previous example.
    tokenizer.pre_tokenizer = Split(Regex(re.escape(cfg.delimiter)), behavior="removed")

    # 3. Set up the Trainer
    trainer = WordLevelTrainer(
        min_frequency=cfg.min_frequency,
    )

    def batch_iterator() -> Iterable[str]:
        for batch in dataset.iter(batch_size=batch_size):
            yield batch[TEXT_COLUMN]

    log.info("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=cfg.model_max_length,
        padding_side=cfg.padding_side,
        truncation_side=cfg.truncation_side,
        pad_token=cfg.special_tokens.pad_token,
    )


@hydra.main(version_base=None, config_path="config", config_name="tokenize")
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
    create_pretraining_dataset(
        meds_dir=meds_data_dir,
        output_dir=raw_data_dir,
        splits={train_split, val_split},
        eoc_token=cfg.tokenization.special_tokens.eoc_token,
        overwrite=cfg.overwrite,
    )

    log.info("Loading raw pretraining dataset as huggingface dataset...")
    dataset = load_pretraining_dataset(
        dataset_dir=raw_data_dir,
        train_split=train_split,
        val_split=val_split,
    )
    log.info(f"Loaded {len(dataset[train_split])} train examples and " f"{len(dataset[val_split])} val examples.")

    log.info("Training custom tokenizer on raw pretraining dataset...")
    tokenizer = train_custom_tokenizer(dataset[train_split], cfg.tokenization, cfg.batch_size)
    tokenizer.save_pretrained(str(output_dir))
    log.info(f"Tokenizer of vocab size {len(tokenizer)} trained")

    log.info("Tokenizing and chunking raw pretraining dataset with the trained tokenizer...")
    tokenized_dataset = dataset.map(
        lambda texts: {
            k: v
            for k, v in tokenizer(
                texts,
                truncation=True,
                stride=cfg.tokenization.stride,
                return_overflowing_tokens=True,
                padding_side=cfg.tokenization.padding_side,
                return_attention_mask=True,
                padding=False,
            ).items()
            if k in ["input_ids", "attention_mask"]
        },
        input_columns=TEXT_COLUMN,
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
    shutil.rmtree(raw_data_dir)
    log.info(f"Tokenized dataset saved to {output_dir / 'tokenized_dataset'}")
