"""Zero-shot LLM inference on natural language EHR datasets with shard-sharding.

Usage:
    uv run llm_inference \
        dataset_dir=data/mimic-iv-nl-dataset \
        split=tuning \
        output_dir=outputs/zero_shot/inference
"""

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

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
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


# def extract_yes_no_logit_probabilities(
#     new_tokens: torch.Tensor,
#     logits: torch.Tensor,
#     pad_token_id: int,
#     yes_token_id: int,
#     no_token_id: int,
# ) -> np.ndarray:
#     """
#     This works on the assumption that the stop string ends with Yes or No
#     So the index of last non-padding token in row of new_tokens is the Yes/No idx.
#     """
#     # logits is a tuple of (B, vocab_size) tensors, one per new token step.
#     # Stack into (B, T_new, vocab_size), then pick the relevant step per sample.
#     stacked_logits = torch.stack(logits, dim=1)  # (B, T_new, vocab_size)

#     # Mask to non-padded tokens
#     new_tokens_mask = new_tokens != pad_token_id  # (B, T_new)

#     if not new_tokens_mask.any(dim=1).all():
#         log.warning("Some sequences are entirely padding tokens")

#     # Find the last non-padded token per sequence: flip so the last True becomes the
#     # 1st then argmax finds it; subtract from (T_new - 1) to convert back to
#     # the original index.
#     yes_no_step_indices = new_tokens_mask.shape[1] - 1 - new_tokens_mask.int().flip(dims=[1]).argmax(dim=1)  # (B,)

#     yes_no_logits = stacked_logits[torch.arange(stacked_logits.shape[0]), yes_no_step_indices]  # (B, vocab_size)]

#     target_logits = yes_no_logits[:, [yes_token_id, no_token_id]]

#     # Softmax over just these two options
#     probs = F.softmax(target_logits, dim=-1)

#     # Return probability of Yes (index 0)
#     return probs[:, 0]


def extract_probability_by_regex(
    completions: list[str], outputs: Any, regex_pattern: re.Pattern, default_prob: float
) -> tuple[list[float], list[float]]:
    probability = [default_prob] * len(completions)
    format_adherence = [0.0] * len(completions)

    for idx, completion in enumerate(completions):
        match = re.search(regex_pattern, completion)
        if match:
            try:
                # Strip whitespace and convert to float
                prob_str = match.group(1).strip()
                probability[idx] = float(prob_str)
                format_adherence[idx] = 1.0
            except (ValueError, IndexError):
                pass  # Keep default probability and format adherence 0 if conversion fails

    return probability, format_adherence


def batch_inference(
    batch_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: DictConfig,
    post_processing_func: Callable[[list[str], Any], tuple[list[float], list[float]]],
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
            tokenizer=tokenizer,  # Pass tokenizer for handling of stopping criteria
            **generation_config,
        )

    # Extract sequences and logits
    new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    probabilities, format_adherence = post_processing_func(completions, outputs)
    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / len(batch_texts)

    return {
        "completion": completions,
        "probability": list(probabilities),
        "format_adherence": list(format_adherence),
        "sample_latency": [sample_latency] * len(batch_texts),
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

    if cfg.post_processing.output_format == "probability":
        regex_pattern = re.compile(cfg.post_processing.regex_pattern)

        def _post_processing_func(completions: list[str], outputs: Any) -> tuple[list[float], list[float]]:
            return extract_probability_by_regex(completions, outputs, regex_pattern, cfg.post_processing.default_prob)

        post_processing_func = _post_processing_func
    else:
        raise ValueError(f"Unsupported output format: {cfg.post_processing.output_format}")

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
                    generation_config=cfg.generation,
                    post_processing_func=post_processing_func,
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
