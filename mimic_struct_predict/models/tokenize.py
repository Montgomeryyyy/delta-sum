import logging
import os
from pathlib import Path
from typing import Any

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
    target_prefix: str = "",
    save_text: bool = False,
) -> dict | None:
    """Tokenize one chunk: prior ``input_field`` texts → input, current ``target_field`` → labels.

    Input is ``<bos> context target_prefix``; labels are ``target <eos>``.
    When ``save_text`` is True, also store ``input_text`` / ``target_text``.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id so the model learns to stop.")
    bos_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    target_text = chunks[chunk_index][target_field].strip()
    if not target_text:
        return None

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
        padding=False,
    )["input_ids"]
    if not target_ids:
        return None
    target_ids = target_ids + [tokenizer.eos_token_id]

    prefix_ids = (
        tokenizer(target_prefix, add_special_tokens=False)["input_ids"] if target_prefix else []
    )

    # At chunk i: input from chunks[:i] (input_field); target from chunks[i] (target_field).
    prev_parts = [
        text for chunk in chunks[:chunk_index] if (text := chunk[input_field].strip())
    ]
    input_text = chunk_join_token.join(prev_parts)
    if not input_text:
        return None

    # Leave room so context+target fit in one causal-LM sequence at train time.
    context_budget = max_length - len(target_ids) - len(bos_ids) - len(prefix_ids)
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
    input_ids = bos_ids + input_ids + prefix_ids

    row = {
        "input_ids": input_ids,
        "labels": list(target_ids),
        "chunk_id": int(chunks[chunk_index].get("chunk_id", chunk_index)),
        "context_len": len(input_ids),
        "target_len": len(target_ids),
    }
    if save_text:
        row["input_text"] = input_text + target_prefix
        row["target_text"] = target_text
    return row


def explode_context_target_examples(
    example: dict,
    input_field: str,
    target_field: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
    chunk_join_token: str,
    target_prefix: str = "",
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
            target_prefix=target_prefix,
            save_text=save_text,
        )
        if row is None:
            continue
        row["subject_id"] = subject_id
        rows.append(row)

    columns = list(chunked_example_features(save_text))
    return {col: [row[col] for row in rows] for col in columns}


def chunked_example_features(save_text: bool) -> datasets.Features:
    """Schema of chunked examples; int32 token ids keep the multi-billion-token dataset small."""
    features = {
        "subject_id": datasets.Value("int64"),
        "chunk_id": datasets.Value("int64"),
        "input_ids": datasets.Sequence(datasets.Value("int32")),
        "labels": datasets.Sequence(datasets.Value("int32")),
        "context_len": datasets.Value("int32"),
        "target_len": datasets.Value("int32"),
    }
    if save_text:
        features["input_text"] = datasets.Value("string")
        features["target_text"] = datasets.Value("string")
    return datasets.Features(features)


def explode_patient_batch(batch: dict, save_text: bool, **kwargs: Any) -> dict:
    """Batched ``map`` fn: patients in → one row per (patient, target chunk) out."""
    out: dict[str, list] = {col: [] for col in chunked_example_features(save_text)}
    for subject_id, chunks in zip(batch["subject_id"], batch["chunks"], strict=True):
        expanded = explode_context_target_examples(
            {"subject_id": subject_id, "chunks": chunks}, save_text=save_text, **kwargs
        )
        for col, values in expanded.items():
            out[col].extend(values)
    return out


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
    target_prefix = str(cfg.tokenization.get("target_prefix", ""))
    max_length = int(cfg.tokenization.max_length)
    save_text = bool(cfg.tokenization.get("save_text", False))

    log.info(
        f"Building chunked examples input_field={input_field!r} target_field={target_field!r} "
        f"target_prefix={target_prefix!r} max_length={max_length} save_text={save_text} → {out_path}..."
    )

    # Stream through map so examples are written to arrow per batch; the full train split is
    # ~1.5M examples / ~14B tokens and does not fit in memory as Python lists.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache_dir = Path(os.environ.get("HF_DATASETS_CACHE") or out_path.parent / ".chunked_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    flat_splits: dict[str, datasets.Dataset] = {}
    for split, rows in source.items():
        flat_splits[split] = rows.map(
            explode_patient_batch,
            batched=True,
            batch_size=cfg.batch_size,
            writer_batch_size=200,
            num_proc=cfg.num_workers if cfg.num_workers > 1 else None,
            remove_columns=rows.column_names,
            features=chunked_example_features(save_text),
            cache_file_name=str(cache_dir / f"chunked_{split}.arrow"),
            load_from_cache_file=False,
            fn_kwargs={
                "save_text": save_text,
                "input_field": input_field,
                "target_field": target_field,
                "tokenizer": tokenizer,
                "max_length": max_length,
                "chunk_join_token": chunk_join_token,
                "target_prefix": target_prefix,
            },
            desc=f"Chunked tokenization [{split}]",
        )
        log.info(f"Split {split}: {len(rows)} patients → {len(flat_splits[split])} chunk examples.")

    tokenized = datasets.DatasetDict(flat_splits)
    n_val = len(tokenized[val_split]) if val_split in tokenized else 0
    log.info(
        f"Saved {len(tokenized[train_split])} train / {n_val} val chunk examples to {out_path}."
    )
    tokenized.save_to_disk(out_path)
    for split_dataset in tokenized.values():
        for cache_file in split_dataset.cache_files:
            Path(cache_file["filename"]).unlink(missing_ok=True)


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
