"""Inference script for structured generative rollout."""

import logging
from pathlib import Path

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from mimic_struct_predict.models.utils import (
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


def batch_embedding_inference(
    ehr_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list]:
    """Run inference on a batch of EHR texts by
    - Tokenizing the input texts with appropriate padding and truncation
    - Generating embeddings of the final token using the model's forward method
    - Return the embeddings as a list of tensors for each example in the batch
    """
    if tokenizer.padding_side != "left":
        raise ValueError("Tokenizer must have left padding for correct embedding extraction.")

    inputs = tokenizer(
        ehr_texts,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Forward pass to get logits for the next token (pass tensors explicitly)
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    # Safe to do take [:, -1, :] as last hidden state as padding is on left
    last_hidden_states = outputs.hidden_states[-1][:, -1, :]

    # Can't do straight to numpy as get 'pyarrow.lib.ArrowInvalid: only handle 1-dimensional arrays'
    return {"embedding": list(last_hidden_states.cpu().float().numpy())}


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_embedding",
)
def main(cfg: DictConfig) -> None:
    """
    For each task:
    1. Check if embeddings dataset already exists for the task and split. If yes, skip to next task.
    2. If not, load the raw EHR text dataset for the task and split.
    3. Run batch inference to generate embeddings for each example and stream results to a new Parquet file.
    """

    log.info("Starting Embedding Inference")
    dataset_dir = Path(cfg.dataset_dir)
    out_dir = Path(cfg.output_dir)
    log.info(f"Output directory: {out_dir}")

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModelForCausalLM,
        model_config=cfg.model,
        eval_mode=True,
        padding_side="left",
    )

    for task_dir in sorted(dataset_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name == ".logs":
            continue

        task_name = task_dir.name
        if cfg.tasks is not None and task_name not in cfg.tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")

        for split in cfg.splits:
            embeddings_path = out_dir / task_name / split / "embeddings.parquet"
            if embeddings_path.exists():
                log.info(
                    f"Embeddings file {embeddings_path} exist for task {task_name} in split {split}."
                    f" Skipping inference."
                )
                continue

            shards = list((task_dir / split).glob("*.parquet"))
            if not shards:
                log.warning(f"No shard files found for task {task_name} in split {split}. Skipping.")
                continue

            dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
            log.info(f"Loaded {len(dataset)} examples from task directory.")
            if len(dataset) == 0:
                raise ValueError(f"Dataset for task {task_name} in split {split} is empty.")

            # Sort to minimize padding within batches
            dataset = dataset.map(
                lambda ehr_text: {"length": len(ehr_text)},
                input_columns=["ehr_text"],
                num_proc=cfg.num_workers,
            )
            dataset = dataset.sort("length", reverse=True)
            dataset = dataset.remove_columns("length")

            writer = None
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"Running inference and streaming to {embeddings_path}")
            try:
                for idx, batch in enumerate(dataset.iter(batch_size=cfg.batch_size)):
                    results = batch_embedding_inference(
                        ehr_texts=batch["ehr_text"],
                        model=model,
                        tokenizer=tokenizer,
                    )
                    for key, values in results.items():
                        batch[key] = values

                    table = pa.Table.from_pydict(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(embeddings_path, table.schema)
                    writer.write_table(table)

                    if (idx + 1) % cfg.log_interval == 0:
                        log.info(f"Processed {idx + 1} batches for task {task_name} in split {split}.")

            except Exception as e:
                log.error(f"Error processing task {task_name} at batch index {idx}: {e}", exc_info=True)
                raise e

            finally:
                # Ensure the Parquet file is properly finalized and closed,
                # even if an exception was raised during the loop.
                if writer is not None:
                    writer.close()
