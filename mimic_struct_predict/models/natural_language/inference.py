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
from typing import Any

import datasets
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithPadding,
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
    inputs: dict[str, torch.Tensor],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    yes_token_id: int,
    no_token_id: int,
    generation_config: DictConfig,
) -> dict[str, list]:
    """Run a single batch of inference and extract answers/probabilities.

    - Runs and times generations
    - Decodes completions
    - Extracts final token logits to compute Yes/No probabilities
    - Computes failure rate based on whether final token is Yes/No
    """

    start_time = time.perf_counter()

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Forward pass to get logits for the next token
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            output_logits=True,
            return_dict_in_generate=True,
            tokenizer=tokenizer,  # Pass tokenizer for handling of stopping criteria
            **generation_config,
        )

    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / inputs["input_ids"].shape[0]

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
    probabilities = compute_yes_probs(final_step_logits, yes_token_id, no_token_id).cpu().numpy()

    # Check if the last token is actually Yes or No for failure rate
    last_token_ids = new_tokens[batch_indices, last_token_indices]
    is_yes = (last_token_ids == yes_token_id).cpu().numpy()
    is_no = (last_token_ids == no_token_id).cpu().numpy()
    format_adherence = is_yes | is_no

    return {
        "completion": completions,
        "probability": [float(p) for p in probabilities],
        "sample_latency": [sample_latency] * inputs["input_ids"].shape[0],
        "format_adherence": format_adherence.tolist(),
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
        - format_adherence: Whether the model's final token was not "Yes" or "No".
    """

    log.info("Starting LLM Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir) / cfg.split
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")

    # 1. Discovery shards: task/split/*.parquet
    # We follow the structure of dataset_dir / task / split / shard.parquet
    selected_tasks = cfg.get("tasks", None)
    task_dirs = sorted(
        task_dir
        for task_dir in dataset_dir.iterdir()
        if task_dir.is_dir() and (selected_tasks is None or task_dir.name in selected_tasks)
    )

    # TODO: Update NL dataset file structure/naming
    if not task_dirs:
        log.error(
            f"No task directories found in {dataset_dir} for "
            f"split {cfg.split} and tasks {selected_tasks if selected_tasks else 'ALL'}"
        )
        return
    log.info(f"Found {len(task_dirs)} task directories to process.")

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

    for task_dir in task_dirs:
        # Find all shard files within the current task directory
        log.info(f"Processing Task: [{task_dir.name}]")
        dataset = datasets.load_from_disk(str(task_dir))
        log.info(f"Loaded {len(dataset)} examples from task directory.")

        try:
            dataset = dataset.map(
                lambda prompts: tokenizer(
                    prompts,
                    # Handled in create_prompts
                    truncation=False,
                    # Will do at batching stage for more efficient dynamic padding
                    padding=False,
                    return_length=True,
                ),
                input_columns=["prompt"],
                batched=True,
                batch_size=cfg.tokenization.batch_size,
            )
            # Sort to minimize padding within batches
            dataset = dataset.sort("length")

            inference_dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])
            collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding="longest",
                pad_to_multiple_of=8,  # For efficient tensor cores if using CUDA
            )
            dataloader = DataLoader(
                inference_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collator,
            )

            # 3d. Batch Inference via DataLoader
            log.info(f"Running inference on {len(dataset)} examples...")
            results: dict[str, list[Any]] = {
                "completion": [],
                "probability": [],
                "sample_latency": [],
                "format_adherence": [],
            }
            for batch in tqdm(dataloader, desc="Inference"):
                batch_results = batch_inference(
                    inputs=batch,
                    model=model,
                    tokenizer=tokenizer,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    generation_config=cfg.generation,
                )

                for key in results:
                    results[key].extend(batch_results[key])

            # 5. Combine results and save to output directory
            for key, values in results.items():
                dataset = dataset.add_column(key, values)

            # Remove internal columns
            cols_to_drop = ["length", "input_ids", "attention_mask"]
            dataset = dataset.remove_columns(cols_to_drop)

            out_file_path = out_dir / cfg.split / task_dir.name
            dataset.save_to_disk(out_file_path)
            log.info(f"Saved results to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing task {task_dir.name}: {e}")
            raise

    log.info("LLM Inference Complete!")
