"""Inference script for structured generative rollout."""

import logging
from pathlib import Path

import datasets
import hydra
import joblib
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm
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


def create_embeddings_dataset(
    output_dir: Path,
    task_dir: Path,
    split: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_workers: int,
    batch_size: int,
) -> None:
    """
    For a given task and split:
    1. Load the raw EHR text dataset for the task and split.
    2. Run batch inference to generate embeddings for each example and stream results to a new
    Parquet file in the output directory.
    """
    shards = (task_dir / split).glob("*.parquet")
    if not shards:
        raise FileNotFoundError(f"No shard files found for task {task_dir.name} in split {split}.")

    dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
    log.info(f"Loaded {len(dataset)} examples from task directory.")
    if len(dataset) == 0:
        raise ValueError(f"Dataset for task {task_dir.name} in split {split} is empty.")

    # Sort to minimize padding within batches
    dataset = dataset.map(
        lambda ehr_text: {"length": len(ehr_text)},
        input_columns=["ehr_text"],
        num_proc=num_workers,
    )
    dataset = dataset.sort("length", reverse=True)
    dataset = dataset.remove_columns("length")

    writer = None
    embeddings_path = get_embedding_path(output_dir, task_dir, split)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Running inference and streaming to {embeddings_path}")
    try:
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
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

    except Exception as e:
        log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)
        raise e

    finally:
        # Ensure the Parquet file is properly finalized and closed,
        # even if an exception was raised during the loop.
        if writer is not None:
            writer.close()


def get_embedding_path(out_dir: Path, task_dir: Path, split: str) -> Path:
    return out_dir / "embeddings" / task_dir.name / split / "embeddings.parquet"


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="train_embedding_lr",
)
def main(cfg: DictConfig) -> None:
    """
    For each task:
    1. Check if embeddings dataset already exists for the task and split. If yes, skip to next task.
    2. If not, load the raw EHR text dataset for the task and split.
    3. Run batch inference to generate embeddings for each example and stream results to a new Parquet file.
    4. Load the generated embeddings dataset for the training and validation splits.
    5. Train a logistic regression model on the training embeddings and evaluate on the validation set, logging the AUC.
    6. Save the trained logistic regression model to disk.
    """

    log.info("Starting Structured Generative Rollout Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir)
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(cfg.output_dir)
    lr_out_dir = Path(hydra_cfg.run.dir) / "models"
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
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        train_path = get_embedding_path(out_dir, task_dir, cfg.split_names.train)
        val_path = get_embedding_path(out_dir, task_dir, cfg.split_names.val)

        for split, split_path in [(cfg.split_names.train, train_path), (cfg.split_names.val, val_path)]:
            if split_path.exists():
                log.info(f"Embeddings already exist for task {task_name} in split {split}. Skipping inference.")
                continue
            else:
                try:
                    create_embeddings_dataset(
                        output_dir=out_dir,
                        task_dir=task_dir,
                        split=split,
                        model=model,
                        tokenizer=tokenizer,
                        num_workers=cfg.num_workers,
                        batch_size=cfg.batch_size,
                    )
                except Exception as e:
                    log.error(
                        f"Failed to create embeddings dataset for task {task_name} in split {split} "
                        f"so not training LR model: {e}",
                        exc_info=True,
                    )
                    if split_path.exists():
                        split_path.unlink()
                    continue

        if not train_path.exists() or not val_path.exists():
            log.warning(
                f"Embeddings dataset not found for task {task_name} in train or val split after inference step. "
                f"Skipping LR training for this task."
            )
            continue

        train_df = pl.read_parquet(train_path, columns=["embedding", "boolean_value"])
        val_df = pl.read_parquet(val_path, columns=["embedding", "boolean_value"])

        dim = len(train_df["embedding"][0])
        train_X, train_y = train_df["embedding"].list.to_array(dim).to_numpy(), train_df["boolean_value"].to_numpy()
        val_X, val_y = val_df["embedding"].list.to_array(dim).to_numpy(), val_df["boolean_value"].to_numpy()

        log.info(
            f"Loaded embeddings dataset for task {task_name} with {len(train_X)} training examples and "
            f"{len(val_X)} validation examples."
        )
        log.info("Percentage of positive labels in training set: {:.2f}%".format(train_y.mean() * 100))
        log.info("Percentage of positive labels in validation set: {:.2f}%".format(val_y.mean() * 100))

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        train_X, train_y = shuffle(train_X, train_y, random_state=cfg.random_seed)
        clf.fit(train_X, train_y)

        auc = roc_auc_score(val_y, clf.predict_proba(val_X)[:, 1])
        log.info(f"Validation AUC for task {task_name}: {auc:.4f}")

        model_path = lr_out_dir / task_name / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)
        log.info(f"Model successfully saved to {model_path}")

    log.info("Inference Complete!")
