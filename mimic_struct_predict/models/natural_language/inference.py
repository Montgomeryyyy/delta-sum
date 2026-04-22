"""Zero-shot LLM inference on natural language EHR datasets with shard-sharding.

Usage:
    uv run llm_inference \
        dataset_dir=data/mimic-iv-nl-dataset \
        split=tuning \
        output_dir=outputs/zero_shot/inference
"""

import logging
import time
from pathlib import Path

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mimic_struct_predict.models.utils import (
    get_token_id,
    load_model_and_tokenizer,
)
from mimic_struct_predict.models.utils.metrics import compute_yes_probs

log = logging.getLogger(__name__)


def batch_inference(
    batch_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    yes_token_id: int,
    no_token_id: int,
    generation_config: DictConfig,
) -> dict[str, list]:
    """Run inference on a batch of texts by
    - Tokenizing the input texts with appropriate padding and truncation
    - Generating continuations using the model's generate method
    - Post-processing the generated tokens to extract compute probabilities,
    and check format adherence
    - Measuring latency for the batch and computing average latency per sample
    """
    start_time = time.perf_counter()

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        max_length=tokenizer.model_max_length - generation_config.max_new_tokens,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Forward pass to get logits for the next token
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            output_logits=True,
            return_dict_in_generate=True,
            tokenizer=tokenizer,  # Pass tokenizer for handling of stopping criteria
            **generation_config,
        )

    # Extract sequences and logits
    new_tokens = outputs.sequences[:, inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # Completions can be differing lengths, so need to handle extracting the final token logits carefully
    # Create mask for non-padded tokens in new_tokens
    new_tokens_mask = new_tokens != tokenizer.pad_token_id

    # Indices of last non-padded tokens per row
    last_token_indices = new_tokens_mask.int().sum(dim=1) - 1
    last_token_indices = last_token_indices.clamp(min=0)

    # stacked_logits: (B, T_generated, V)
    stacked_logits = torch.stack(outputs.logits, dim=1)

    # Extract logits for the last token of each sample in the batch
    batch_indices = torch.arange(inputs["input_ids"].shape[0], device=model.device)
    final_step_logits = stacked_logits[batch_indices, last_token_indices]  # (B, V)

    # Compute probability of Yes vs No at the final step
    probabilities = compute_yes_probs(final_step_logits, yes_token_id, no_token_id).cpu().tolist()

    # Check if the last token is actually Yes or No for failure rate
    last_token_ids = new_tokens[batch_indices, last_token_indices]
    format_adherence = (last_token_ids == yes_token_id) | (last_token_ids == no_token_id)
    format_adherence = format_adherence.cpu().tolist()

    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / len(batch_texts)

    del outputs, stacked_logits, final_step_logits, last_token_ids, new_tokens_mask

    return {
        "completion": completions,
        "probability": probabilities,
        "sample_latency": [sample_latency] * len(batch_texts),
        "format_adherence": format_adherence,
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_direct",
)
def main(cfg: DictConfig) -> None:
    """
    Run zero-shot LLM inference on natural language EHR datasets with shard-sharding.
    Hydra config controls dataset location, model/config selection, and generation parameters.

    Steps:
    1. Discover shards based on dataset_dir, split, and optional task filtering.
    2. Load model and tokenizer
    3. For each shard:
        a. Load shard data
        b. Create prompts from EHR text and task descriptions.
        c. Tokenize prompts and prepare DataLoader.
        d. Run batch inference to get completions and probabilities.
        e. Combine results and save to output directory.

    Returns dataset with columns:
        - subject_id: The patient identifier.
        - prediction_id: The unique identifier for the prediction instance.
        - task: The clinical prediction task (e.g. mortality, length of stay).
        - boolean_label: The ground truth label for the prediction task.
        - prompt: The generated prompt for the model.
        - completion: The model's generated completion.
        - probability: The computed probability of "Yes" at the final token.
        - sample_latency: Time taken for generation per sample.
        - format_adherence: Whether the model's final token was is expected yes or no token.
    """

    log.info("Starting LLM Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir) / cfg.split
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModelForCausalLM,
        model_config=cfg.model,
        eval_mode=True,
        padding_side="left",
    )
    yes_token_id = get_token_id(tokenizer, cfg.tokenization.yes_token)
    no_token_id = get_token_id(tokenizer, cfg.tokenization.no_token)
    log.info(f"Yes token ID: {yes_token_id}, No token ID: {no_token_id}")

    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        dataset = datasets.load_from_disk(str(task_dir))
        log.info(f"Loaded {len(dataset)} examples from task directory.")

        # Sort to minimize padding within batches
        dataset = dataset.map(
            lambda batch: {"length": [len(prompt) for prompt in batch]},
            input_columns=["prompt"],
            batched=True,
            num_proc=cfg.num_workers,
        )
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns("length")

        # Batch wise inference and streaming results to Parquet to avoid memory issues with large datasets
        # and allow for partial results to be saved even if the process is interrupted.
        writer = None
        out_file_path = out_dir / cfg.split / task_dir.name / "results.parquet"
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Running inference and streaming to {out_file_path}")
        try:
            for i in tqdm(range(0, len(dataset), cfg.batch_size)):
                batch = dataset[i : i + cfg.batch_size]
                results = batch_inference(
                    batch_texts=batch["prompt"],
                    model=model,
                    tokenizer=tokenizer,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    generation_config=cfg.generation,
                )
                for key, values in results.items():
                    batch[key] = values

                table = pa.Table.from_pydict(batch)
                if writer is None:
                    writer = pq.ParquetWriter(out_file_path, table.schema)
                writer.write_table(table)

        except Exception as e:
            log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)

        finally:
            # Ensure the Parquet file is properly finalized and closed,
            # even if an exception was raised during the loop.
            if writer is not None:
                writer.close()

    log.info("Inference Complete!")
