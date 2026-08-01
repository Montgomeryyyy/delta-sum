"""Inference script for structured generative rollout."""

import logging
import math
import typing
from pathlib import Path

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def run_inference_and_save(
    dataset: datasets.Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    result_path: Path,
    task_name: str,
    split: str,
) -> None:
    """Run batch inference on the dataset and stream results to a Parquet file."""
    writer = None
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Running inference and streaming to {result_path}")
    try:
        num_batches = math.ceil(len(dataset) / batch_size)
        for idx, batch in enumerate(
            tqdm(
                dataset.iter(batch_size=batch_size),
                total=num_batches,
                desc=f"Inferencing {task_name}/{split}",
            )
        ):
            inputs = tokenizer.pad(
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state[:, -1, :]
            embeddings = list(last_hidden_states.cpu().float().numpy())

            output_batch = {
                "ehr_text": batch["ehr_text"],
                "embedding": embeddings,
                "boolean_value": batch["boolean_value"],
            }

            table = pa.Table.from_pydict(output_batch)
            if writer is None:
                writer = pq.ParquetWriter(result_path, table.schema)
            writer.write_table(table)

    except Exception as e:
        log.error(f"Error processing task {task_name} at batch index {idx}: {e}", exc_info=True)
        raise e

    finally:
        # Ensure the Parquet file is properly finalized and closed,
        # even if an exception was raised during the loop.
        if writer is not None:
            writer.close()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_embedding",
)
def main(cfg: DictConfig) -> None:
    """
    For the given task:
    1. Check if embeddings dataset already exists for the task and split. If yes and not overwrite skip to next split.
    2. If not, load the raw EHR text dataset for the task and split.
    3. Run tokenization and sort by length
    4. Run batch inference to generate embeddings for each example and stream results to a new Parquet file.
    """
    log.info("Starting Embedding Inference")
    dataset_dir = Path(cfg.dataset_dir)
    out_dir = Path(cfg.output_dir)
    log.info(f"Output directory: {out_dir}")

    task_dir = dataset_dir / cfg.task_name
    if not task_dir.is_dir():
        raise ValueError(f"Task directory {task_dir} does not exist.")

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")

    model = AutoModel.from_pretrained(**cfg.model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(padding_side="left", truncation_side="left", **cfg.tokenizer)

    log.info(f"Processing Task: [{cfg.task_name}]")

    for split in cfg.splits:
        embeddings_path = out_dir / cfg.task_name / split / "embeddings.parquet"
        if not cfg.overwrite and embeddings_path.exists():
            log.info(
                f"Embeddings file {embeddings_path} exist for task {cfg.task_name} in split {split}."
                f" Skipping inference."
            )
            continue

        shards = list((task_dir / split).glob("*.parquet"))
        if not shards:
            log.warning(f"No shard files found for task {cfg.task_name} in split {split}. Skipping.")
            continue

        dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
        log.info(f"Loaded {len(dataset)} examples from task directory.")
        if len(dataset) == 0:
            raise ValueError(f"Dataset for task {cfg.task_name} in split {split} is empty.")
        try:

            def tokenize_fn(batch: dict) -> typing.Any:
                # Pre-slice to save tokenizer compute
                if tokenizer.model_max_length:
                    char_limit = tokenizer.model_max_length * 10
                    texts = [t[-char_limit:] for t in batch["ehr_text"]]

                tokenized = tokenizer(
                    texts,
                    truncation=True,
                    padding=False,  # Pad at inference time
                )
                # Overwrite ehr_text with the truncated version so we don't save what we actually infer on
                tokenized["ehr_text"] = texts
                # Add a length column for sorting
                tokenized["token_len"] = [len(ids) for ids in tokenized["input_ids"]]
                return tokenized

            dataset = dataset.map(
                tokenize_fn,
                batched=True,
                num_proc=cfg.num_workers,
                desc="Tokenizing and truncating",
            )
            dataset = dataset.sort("token_len", reverse=True)
            dataset = dataset.remove_columns("token_len")

            run_inference_and_save(
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                batch_size=cfg.batch_size,
                result_path=embeddings_path,
                task_name=cfg.task_name,
                split=split,
            )
        finally:
            # Clean up intermediate and final cache files to prevent disk explosion
            dataset.cleanup_cache_files()
            for cache_file in dataset.cache_files:
                if "filename" in cache_file:
                    cache_path = Path(cache_file["filename"])
                    if cache_path.exists():
                        cache_path.unlink()
