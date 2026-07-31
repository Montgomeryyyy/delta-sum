"""Inference script for structured embedding and logistic regression."""

import logging
import time
from pathlib import Path

import datasets
import hydra
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizerBase

from mimic_struct_predict.models.embedding_and_lr.inference_embedding_only import batch_embedding_inference
from mimic_struct_predict.models.utils import (
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


def batch_embedding_and_lr_inference(
    ehr_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    lr_pipeline: Pipeline,
    max_length: int | None,
) -> dict[str, list]:
    """Run inference on a batch of EHR texts by
    - Tokenizing the input texts with appropriate padding and truncation
    - Generating embeddings of the final token using the model's forward method
    - Return the embeddings as a list of tensors for each example in the batch
    """
    start_time = time.perf_counter()

    embeddings = batch_embedding_inference(
        ehr_texts=ehr_texts,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
    )["embedding"]

    # Get probability of positive class from logistic regression model
    lr_predictions = lr_pipeline.predict_proba(embeddings)[:, 1].tolist()

    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / len(ehr_texts)

    # Can't do straight to numpy as get 'pyarrow.lib.ArrowInvalid: only handle 1-dimensional arrays'
    return {
        "probability": lr_predictions,
        "sample_latency": [sample_latency] * len(ehr_texts),
        # Can't have incorrect format output for embeddings (if inference completes)
        "format_adherence": [1] * len(ehr_texts),
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference",
)
def main(cfg: DictConfig) -> None:
    """
    1. Load pre-trained transformer model and tokenizer
    For each task:
        2. Load the trained logistic regression model and input prompts for the task
        2. Sort the dataset by input length to minimize padding within batches
        3. Run batch inference to generate embeddings for each example and stream results to a new Parquet file.
    """

    log.info("Starting Embedding + Logistic Regression Inference")
    dataset_dir = Path(cfg.dataset_dir)
    lr_model_dir = Path(cfg.lr_model_dir)
    out_dir = Path(HydraConfig.get().run.dir)

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModel,
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

        # Sort by modification time, most recent last
        saved_lr_models = sorted(
            (lr_model_dir / task_name).glob("multirun_*/best_trial/model.joblib"), key=lambda p: p.stat().st_mtime
        )
        if not saved_lr_models:
            log.warning(
                f"No best_trial logistic regression model found for task {task_name} in {lr_model_dir / task_name}."
                f" Skipping."
            )
            continue
        if len(saved_lr_models) > 1:
            log.warning(
                f"Multiple model training runs found for task {task_name} in {lr_model_dir / task_name}."
                f" Using the one with most recent timestamp: {saved_lr_models[-1]}"
            )

        log.info(f"Loading logistic regression model from {saved_lr_models[-1]}")
        lr_pipeline = joblib.load(saved_lr_models[-1])

        shards = list((task_dir / cfg.split).glob("*.parquet"))
        if not shards:
            log.warning(f"No shard files found for task {task_name} in split {cfg.split}. Skipping.")
            continue

        dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
        log.info(f"Loaded {len(dataset)} examples from task directory.")
        if len(dataset) == 0:
            raise ValueError(f"Dataset for task {task_name} in split {cfg.split} is empty.")

        # Sort to minimize padding within batches
        dataset = dataset.map(
            lambda ehr_text: {"length": len(ehr_text)},
            input_columns=["ehr_text"],
            num_proc=cfg.num_workers,
        )
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns("length")

        writer = None
        result_dir = out_dir / cfg.split / task_name / "results.parquet"
        result_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Running inference and streaming to {result_dir}")
        try:
            for idx, batch in enumerate(dataset.iter(batch_size=cfg.batch_size)):
                results = batch_embedding_and_lr_inference(
                    ehr_texts=batch["ehr_text"],
                    model=model,
                    lr_pipeline=lr_pipeline,
                    tokenizer=tokenizer,
                    max_length=cfg.max_length,
                )
                for key, values in results.items():
                    batch[key] = values

                table = pa.Table.from_pydict(batch)
                if writer is None:
                    writer = pq.ParquetWriter(result_dir, table.schema)
                writer.write_table(table)

                if (idx + 1) % cfg.log_interval == 0:
                    log.info(f"Processed {idx + 1} batches for task {task_name} in split {cfg.split}.")

        except Exception as e:
            log.error(f"Error processing task {task_name} at batch index {idx}: {e}", exc_info=True)
            raise e

        finally:
            # Ensure the Parquet file is properly finalized and closed,
            # even if an exception was raised during the loop.
            if writer is not None:
                writer.close()
