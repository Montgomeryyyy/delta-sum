"""Inference script for generating embeddings from EHR text."""

import logging
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def get_collate_fn(tokenizer: PreTrainedTokenizerBase) -> Callable:
    """Returns a collate function that pads tokenized inputs."""

    def collate_fn(batch_list: list[dict]) -> dict:
        batch_dict = {
            "input_ids": [item["input_ids"] for item in batch_list],
            "attention_mask": [item["attention_mask"] for item in batch_list],
        }
        inputs = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
        return {
            "inputs": inputs,
            "ehr_text": [item["ehr_text"] for item in batch_list],
            "boolean_value": [item["boolean_value"] for item in batch_list],
        }

    return collate_fn


def get_tokenize_fn(tokenizer: PreTrainedTokenizerBase) -> Callable:
    """Returns a tokenization function for dataset mapping."""
    char_limit = (tokenizer.model_max_length * 10) if tokenizer.model_max_length else None

    def tokenize_fn(batch: dict) -> Any:
        texts = batch["ehr_text"]
        if char_limit:
            texts = [t[-char_limit:] for t in texts]

        tokenized = tokenizer(texts, truncation=True, padding=False)
        tokenized["ehr_text"] = texts
        tokenized["token_len"] = [len(ids) for ids in tokenized["input_ids"]]
        return tokenized

    return tokenize_fn


def flush_writer(
    writer: pq.ParquetWriter | None,
    accumulated_batches: dict,
    result_path: Path,
) -> pq.ParquetWriter:
    """Writes accumulated batches to Parquet and returns the writer."""
    table = pa.Table.from_pydict(accumulated_batches)
    if writer is None:
        writer = pq.ParquetWriter(result_path, table.schema)
    writer.write_table(table)
    return writer


def run_inference_and_save(
    dataset: datasets.Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
    write_batch_size: int,
    result_path: Path,
    task_name: str,
    split: str,
) -> None:
    """Run batch inference on the dataset and stream results to a Parquet file."""
    log.info(f"Running inference and streaming to {result_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(tokenizer),
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    writer = None
    accumulated_batches: dict[str, list] = {"ehr_text": [], "embedding": [], "boolean_value": []}
    num_accumulated = 0

    try:
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Inferencing {task_name}/{split}")):
            inputs = batch["inputs"].to(model.device)

            with torch.inference_mode():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state[:, -1, :]
            embeddings = list(last_hidden_states.cpu().float().numpy())

            accumulated_batches["ehr_text"].extend(batch["ehr_text"])
            accumulated_batches["embedding"].extend(embeddings)
            accumulated_batches["boolean_value"].extend(batch["boolean_value"])
            num_accumulated += 1

            if num_accumulated >= write_batch_size:
                writer = flush_writer(writer, accumulated_batches, result_path)
                accumulated_batches = {"ehr_text": [], "embedding": [], "boolean_value": []}
                num_accumulated = 0

        if num_accumulated > 0:
            writer = flush_writer(writer, accumulated_batches, result_path)

    except Exception as e:
        log.error(f"Error processing task {task_name} at batch index {idx}: {e}", exc_info=True)
        raise e
    finally:
        if writer is not None:
            writer.close()


def process_split(
    split: str,
    cfg: DictConfig,
    task_dir: Path,
    out_dir: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Process a single split for the given task."""
    embeddings_path = out_dir / cfg.task_name / split / "embeddings.parquet"
    if not cfg.overwrite and embeddings_path.exists():
        log.info(f"Embeddings file {embeddings_path} exist for task {cfg.task_name} in split {split}. Skipping.")
        return
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    shards = list((task_dir / split).glob("*.parquet"))
    if not shards:
        log.warning(f"No shard files found for task {cfg.task_name} in split {split}. Skipping.")
        return

    dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
    log.info(f"Loaded {len(dataset)} examples from {split} split.")

    if len(dataset) == 0:
        raise ValueError(f"Dataset for task {cfg.task_name} in split {split} is empty.")

    try:
        dataset = dataset.map(
            get_tokenize_fn(tokenizer),
            batched=True,
            num_proc=cfg.num_workers,
            desc=f"Tokenizing {split}",
        )
        dataset = dataset.sort("token_len", reverse=True)
        dataset = dataset.remove_columns("token_len")

        run_inference_and_save(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            write_batch_size=cfg.write_batch_size,
            result_path=embeddings_path,
            task_name=cfg.task_name,
            split=split,
        )
    finally:
        dataset.cleanup_cache_files()
        for cache_file in dataset.cache_files:
            if "filename" in cache_file:
                cache_path = Path(cache_file["filename"])
                if cache_path.exists():
                    cache_path.unlink()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_embedding",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for generating embeddings."""
    log.info("Starting Embedding Inference")
    dataset_dir = Path(cfg.dataset_dir)
    out_dir = Path(cfg.output_dir)

    log.info(f"Output directory: {out_dir}")

    task_dir = dataset_dir / cfg.task_name
    if not task_dir.is_dir():
        raise ValueError(f"Task directory {task_dir} does not exist.")

    log.info("Loading model and tokenizer...")
    model = AutoModel.from_pretrained(**cfg.model)
    model.eval()

    if cfg.get("compile_model", True):
        log.info("Compiling model for faster inference...")
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(padding_side="left", truncation_side="left", **cfg.tokenizer)

    log.info(f"Processing Task: [{cfg.task_name}]")

    for split in cfg.splits:
        process_split(split, cfg, task_dir, out_dir, model, tokenizer)


if __name__ == "__main__":
    main()
