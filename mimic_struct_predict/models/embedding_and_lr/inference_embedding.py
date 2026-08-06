"""Inference script for generating embeddings from EHR text."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

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


def get_collate_fn(
    tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: int | None, dataset_columns: list[str]
) -> Callable:
    """Returns a collate function that pads tokenized inputs."""
    columns_to_keep = [col for col in dataset_columns if col != "input_ids"]

    def collate_fn(batch_list: list[dict]) -> dict:
        batch_dict = {
            "input_ids": [item["input_ids"] for item in batch_list],
        }
        inputs = tokenizer.pad(batch_dict, padding=True, return_tensors="pt", pad_to_multiple_of=pad_to_multiple_of)

        inputs["attention_mask"] = (inputs["input_ids"] != tokenizer.pad_token_id).long()

        result = {"inputs": inputs}
        for col in columns_to_keep:
            result[col] = [item[col] for item in batch_list]
        return result

    return collate_fn


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
    pad_to_multiple_of: int | None,
) -> None:
    """Run batch inference on the dataset and stream results to a Parquet file."""
    log.info(f"Running inference and streaming to {result_path}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=get_collate_fn(tokenizer, pad_to_multiple_of, dataset.column_names),
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    writer = None
    accumulated_batches = defaultdict(list)
    num_accumulated = 0

    try:
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Embedding Inference {task_name}/{split}", mininterval=30)):
            inputs = batch["inputs"].to(model.device)

            with torch.inference_mode():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state[:, -1, :]
            embeddings = list(last_hidden_states.cpu().float().numpy())

            for col in dataset.column_names:
                if col != "input_ids":
                    accumulated_batches[col].extend(batch[col])
            accumulated_batches["embedding"].extend(embeddings)
            num_accumulated += 1

            if num_accumulated >= write_batch_size:
                log.info(f"Writing {num_accumulated} batches to file")
                writer = flush_writer(writer, accumulated_batches, result_path)
                accumulated_batches.clear()
                num_accumulated = 0

        if num_accumulated > 0:
            log.info(f"Writing {num_accumulated} batches to file")
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

    dataset_path = task_dir / split
    if not dataset_path.exists() or not (dataset_path / "dataset_info.json").exists():
        log.warning(
            f"No tokenized dataset found for task {cfg.task_name} in split {split} at {dataset_path}. Skipping."
        )
        return

    dataset = datasets.Dataset.load_from_disk(str(dataset_path))
    log.info(f"Loaded {len(dataset)} examples from {split} split.")

    if len(dataset) == 0:
        raise ValueError(f"Dataset for task {cfg.task_name} in split {split} is empty.")

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
        pad_to_multiple_of=cfg.pad_to_multiple_of,
    )


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

    log.info(f"Loading model and tokenizer ({cfg.model.pretrained_model_name_or_path})...")
    model = AutoModel.from_pretrained(**cfg.model)
    model.eval()

    if cfg.get("compile_model", True):
        log.info("Compiling model for faster inference...")
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(padding_side="left", truncation_side="left", **cfg.tokenizer)

    log.info(f"Processing Task: [{cfg.task_name}]")

    for split in cfg.splits:
        process_split(split, cfg, task_dir, out_dir, model, tokenizer)


if __name__ == "__main__":
    main()
