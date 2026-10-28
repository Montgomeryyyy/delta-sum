import logging
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig
from transformers import (
    PreTrainedTokenizerFast,
)

from mimic_struct_predict.models.utils.model_loading import load_tokenizer

log = logging.getLogger(__name__)


def load_parquet_dataset(input_dir: Path, train_split: str, val_split: str) -> datasets.DatasetDict:
    log.info(f"Loading parquet dataset from {input_dir}...")
    datafiles: dict[str, list[str]] = {train_split: [], val_split: []}
    for shard_path in input_dir.glob("*/*.parquet"):
        split = shard_path.parent.name
        if split not in {train_split, val_split}:
            continue
        datafiles[split].append(str(shard_path))
    return datasets.load_dataset("parquet", data_files=datafiles)


def load_summarized_dataset(input_dir: Path, train_split: str) -> datasets.DatasetDict:
    log.info(f"Loading summarized dataset from {input_dir}...")
    dataset = datasets.load_from_disk(str(input_dir))
    if isinstance(dataset, datasets.Dataset):
        return datasets.DatasetDict({train_split: dataset})
    return dataset


def tokenize_chunk(
    chunks: list[dict],
    chunk_index: int,
    input_field: str,
    target_field: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    chunk_join_token: str,
    save_text: bool = False,
) -> dict | None:
    """Tokenize one chunk: prior ``input_field`` texts → input, current ``target_field`` → labels.

    When ``save_text`` is True, also store ``input_text`` / ``target_text``.
    """
    target_text = chunks[chunk_index][target_field].strip()
    if not target_text:
        return None

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    if not target_ids:
        return None

    # At chunk i: input from chunks[:i] (input_field); target from chunks[i] (target_field).
    prev_parts = [
        text for chunk in chunks[:chunk_index] if (text := chunk[input_field].strip())
    ]
    input_text = chunk_join_token.join(prev_parts)
    if not input_text:
        return None

    # Leave room so context+target fit in one causal-LM sequence at train time.
    context_budget = max_length - len(target_ids)
    if context_budget <= 0:
        return None

    input_ids = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=True,
        truncation_side="left",
        max_length=context_budget,
        padding=False,
    )["input_ids"]
    if not input_ids:
        return None

    row = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(target_ids),
        "chunk_id": int(chunks[chunk_index].get("chunk_id", chunk_index)),
        "context_len": len(input_ids),
        "target_len": len(target_ids),
    }
    if save_text:
        row["input_text"] = input_text
        row["target_text"] = target_text
    return row


def explode_context_target_examples(
    example: dict,
    input_field: str,
    target_field: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    chunk_join_token: str,
    save_text: bool = False,
) -> dict:
    """Expand one patient into one supervised example per chunk with a target."""
    rows: list[dict] = []
    chunks = example["chunks"]
    subject_id = example["subject_id"]
    for chunk_index in range(len(chunks)):
        row = tokenize_chunk(
            chunks,
            chunk_index,
            input_field=input_field,
            target_field=target_field,
            tokenizer=tokenizer,
            max_length=max_length,
            chunk_join_token=chunk_join_token,
            save_text=save_text,
        )
        if row is None:
            continue
        row["subject_id"] = subject_id
        rows.append(row)

    columns = [
        "subject_id",
        "chunk_id",
        "input_ids",
        "attention_mask",
        "labels",
        "context_len",
        "target_len",
    ]
    if save_text:
        columns.extend(["input_text", "target_text"])

    if not rows:
        return {col: [] for col in columns}

    return {col: [row[col] for row in rows] for col in columns}


def tokenize_patient_dataset(
    dataset: datasets.DatasetDict,
    text_field: str,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DictConfig,
    out_path: Path,
) -> None:
    """Classic flat tokenization with overflowing windows → ``tokenized_dataset``."""
    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val
    stride = int(cfg.tokenization.get("stride", 128))

    log.info(f"Tokenizing {text_field!r} → {out_path}...")
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
                stride=stride,
            ).items()
            if k in ["input_ids", "attention_mask"]
        },
        input_columns=text_field,
        remove_columns=dataset[train_split].column_names,
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
    )

    n_val = len(tokenized_dataset[val_split]) if val_split in tokenized_dataset else 0
    log.info(
        f"Tokenized {text_field!r}: {len(tokenized_dataset[train_split])} train / {n_val} val examples."
    )
    tokenized_dataset.save_to_disk(out_path)
    log.info(f"Saved tokenized dataset to {out_path}")
    tokenized_dataset.cleanup_cache_files()


def tokenize_chunked_dataset(
    source: datasets.DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DictConfig,
    out_path: Path,
) -> None:
    """Per-chunk context→target examples; flat dataset for causal LM with masked labels."""
    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val
    input_field = str(cfg.tokenization.input_field)
    target_field = str(cfg.tokenization.target_field)
    chunk_join_token = str(cfg.tokenization.get("chunk_join_token", "\n\n"))
    max_length = int(cfg.tokenization.max_length)
    save_text = bool(cfg.tokenization.get("save_text", False))

    log.info(
        f"Building chunked examples input_field={input_field!r} target_field={target_field!r} "
        f"max_length={max_length} save_text={save_text} → {out_path}..."
    )

    flat_splits: dict[str, datasets.Dataset] = {}
    for split, rows in source.items():
        examples: list[dict] = []
        for example in rows:
            expanded = explode_context_target_examples(
                example,
                input_field=input_field,
                target_field=target_field,
                tokenizer=tokenizer,
                max_length=max_length,
                chunk_join_token=chunk_join_token,
                save_text=save_text,
            )
            n = len(expanded["input_ids"])
            for i in range(n):
                examples.append({k: expanded[k][i] for k in expanded})
        flat_splits[split] = datasets.Dataset.from_list(examples)
        log.info(f"Split {split}: {len(rows)} patients → {len(examples)} chunk examples.")

    tokenized = datasets.DatasetDict(flat_splits)
    n_val = len(tokenized[val_split]) if val_split in tokenized else 0
    log.info(
        f"Saved {len(tokenized[train_split])} train / {n_val} val chunk examples to {out_path}."
    )
    tokenized.save_to_disk(out_path)
    tokenized.cleanup_cache_files()


def check_output_path(out_path: Path, overwrite: bool) -> bool:
    if not overwrite and out_path.exists():
        log.info(
            f"Tokenized dataset already exists at {out_path} and overwrite is "
            f"{overwrite}. Skipping tokenization."
        )
        return True
    return False


def get_tokenizer(cfg: DictConfig) -> PreTrainedTokenizerFast:
    log.info(f"Loading tokenizer {cfg.tokenizer.pre_trained_tokenizer}")
    return load_tokenizer(cfg.tokenizer.pre_trained_tokenizer, padding_side=cfg.tokenizer.padding_side)


@hydra.main(version_base=None, config_path="config/tokenize")
def main(cfg: DictConfig) -> None:
    """
    Tokenize EHR text for pretraining.

    ``chunked=false``: patient-level ``ehr_text`` → ``output_dir/tokenized_dataset``.
    ``chunked=true``: prior ``input_field`` + current ``target_field`` → ``output_dir``
    (``output_dir`` is the HF dataset path itself).
    """
    chunked = bool(cfg.get("chunked", False))
    output_dir = Path(cfg.output_dir)
    input_dir = Path(cfg.input_dir)
    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val

    if chunked:
        out_path = output_dir
        if check_output_path(out_path, cfg.overwrite):
            return

        source = load_summarized_dataset(input_dir, train_split)
        n_val = len(source[val_split]) if val_split in source else 0
        log.info(
            f"Chunked tokenization from {input_dir} "
            f"({len(source[train_split])} train / {n_val} val patients)."
        )
        tokenize_chunked_dataset(source, get_tokenizer(cfg), cfg, out_path)
        return

    # Classic patient-level path.
    out_path = output_dir / "tokenized_dataset"
    if check_output_path(out_path, cfg.overwrite):
        return

    source = load_parquet_dataset(input_dir, train_split, val_split)
    log.info(
        f"Loaded {len(source[train_split])} train and {len(source[val_split])} val patient-level examples."
    )
    tokenize_patient_dataset(source, "ehr_text", get_tokenizer(cfg), cfg, out_path)


if __name__ == "__main__":
    main()
