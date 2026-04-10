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
    create_prompt,
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

    # Completetions can be differing lengths, so need to handle extracting the final token logits carefully
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
    config_name="llm_inference",
)
def run_llm_inference(cfg: DictConfig) -> None:
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
    dataset_dir = Path(cfg.dataset_dir)
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")
    split = cfg.split

    # 1. Discovery shards: task/split/*.parquet
    # We follow the structure of dataset_dir / task / split / shard.parquet
    selected_tasks = cfg.get("tasks", None)
    shards = sorted(
        shard_path
        for shard_path in dataset_dir.glob(f"*/{split}/*.parquet")
        if (selected_tasks is None or shard_path.parent.parent.name in selected_tasks)
        and shard_path.parent.parent.name != "all_events"
    )

    # TODO: Update NL dataset file structure/naming
    if not shards:
        log.error(
            f"No shards found in {dataset_dir} for "
            f"split {split} and tasks {selected_tasks if selected_tasks else 'ALL'}"
        )
        return
    log.info(f"Found {len(shards)} shards to process.")

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModelForCausalLM,
        model_config=cfg,
        eval_mode=True,
        padding_side="left",
    )
    yes_token_id = get_token_id(tokenizer, "Yes")
    no_token_id = get_token_id(tokenizer, "No")
    log.info(f"Yes token ID: {yes_token_id}, No token ID: {no_token_id}")

    for shard_path in shards:
        task = shard_path.parent.parent.name
        shard_name = shard_path.name
        out_file_path = out_dir / task / split / shard_name
        out_file_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(f"Processing Task: [{task}] | Shard: {shard_name}")

        try:
            # 3a. Load shard data (it's not the train dataset but means 1 dataset is returned)
            dataset = datasets.Dataset.from_parquet(str(shard_path))
            log.info(f"Loaded {len(dataset)} examples from shard.")

            try:
                task_description = cfg.task_description[task]
            except KeyError:
                log.error(f"Task description not found for task: {task}")
                raise ValueError(f"Task description not found for task: {task}")

            # 3b. Create prompts from EHR text and task descriptions.
            log.info("Generating prompts...")
            dataset = dataset.map(
                lambda ehr_text: {
                    "prompt": create_prompt(
                        ehr_text=ehr_text,
                        task_description=task_description,
                        tokenizer=tokenizer,
                        enable_thinking=cfg.tokenization.enable_thinking,
                        max_prompt_length=cfg.tokenization.max_prompt_length,
                        max_new_tokens=cfg.generation.get("max_new_tokens", None),
                    )
                },
                input_columns=cfg.get("ehr_text_column", "ehr_text"),
                # Drop ehr_text after creating prompt to save memory, we won't need it anymore and it can be large.
                # remove_columns="ehr_text",
                batched=False,
                desc="Creating prompts",
            )

            # 3c.Tokenize prompts and prepare DataLoader.
            log.info("Tokenizing prompts...")
            dataset = dataset.map(
                lambda prompts: tokenizer(
                    prompts,
                    truncation=None,  # Truncation handled by create_prompt
                    max_length=cfg.tokenization.max_prompt_length,
                    padding="do_not_pad",  # We'll handle padding in the DataLoader with a collator
                    return_length=True,
                ),
                input_columns="prompt",
                batched=True,
                desc="Tokenizing prompts",
            )
            dataset = dataset.sort("length")  # Sort by prompt length for more efficient batching

            inference_dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])
            collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True,
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

            dataset.to_parquet(out_file_path)
            log.info(f"Saved results to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing shard {shard_path}: {e}")
            raise

    log.info("LLM Inference Complete!")


if __name__ == "__main__":
    run_llm_inference()
