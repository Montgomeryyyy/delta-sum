"""Inference script for structured generative rollout evaluation."""

import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Type, cast

import datasets
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteriaList

from mimic_struct_predict.models.structured.stopping_criteria import WordLevelStopCriteria
from mimic_struct_predict.models.utils import (
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


def extract_last_category_content(text: str, category_prefix: str, delimiter: str) -> str:
    pattern = rf"{category_prefix}(.*?){delimiter}"
    matches = re.findall(pattern, text)
    if matches:
        return str(matches[-1])  # Return the last extracted bit
    log.warning("No category header found in the text. This may indicate a formatting issue.")
    return "Unknown"


def parse_category_code(line: str, category_prefix: str) -> str:
    """Extracts the category code from a header line. Expected format is '[CATEGORY PREFIX][code]'."""
    return line[len(category_prefix) :]


def parse_time_line(line: str, time_prefix: str, time_unspecified: str, delimiter: str) -> tuple[int, int] | None:
    """Extracts hours and minutes from a time line.
    Expected format is '[TIME PREFIX][HH]h//[MM]m' or '[TIME PREFIX]Unspecified'.
    Returns a tuple of (hours, minutes) or None if the format is invalid.
    """
    time_content = line[len(time_prefix) :]
    if time_content == time_unspecified:
        return 23, 59  # Default to end of day if time is unspecified
    try:
        parts = time_content.split(delimiter)
        hours = int(parts[0].lower().replace("h", ""))
        minutes = int(parts[1].lower().replace("m", ""))
        if not (0 <= hours <= 23 and 0 <= minutes <= 59):
            log.warning(f"Invalid time range: {time_content}. Keeping current time unchanged.")
            return None
        return hours, minutes
    except (ValueError, IndexError):
        log.warning(f"Invalid time format: {time_content}. Keeping current time unchanged.")
        return None


def parse_delta_line(line: str, delta_prefix: str) -> int:
    """Extracts the number of days from a delta line. Expected format is '[DELTA PREFIX][+][N] days'."""
    delta_content = line[len(delta_prefix) :]
    try:
        days_str = delta_content.split(" ")[0]
        delta_days = int(days_str.lstrip("+"))
        return delta_days
    except (ValueError, IndexError):
        log.warning(f"Invalid delta format: {delta_content}. No time delta will be applied.")
        return 0


def parse_completion(
    text: str,
    prediction_time: datetime,
    prediction_header: str,
    end_time_delta: timedelta | None,
    end_codes: list[str] | None,
    tokenization_cfg: DictConfig,
) -> tuple[list[datetime], list[str], bool, bool]:
    # Initialize current_time to the start of the provided prediction_time
    current_time = prediction_time
    header = prediction_header
    prefixes = tokenization_cfg.prefixes
    prediction_end_time = prediction_time + end_time_delta if end_time_delta is not None else datetime.max
    end_codes = end_codes if end_codes is not None else []

    found, finished = False, False
    timestamps, codes = [], []

    for line in text.split(tokenization_cfg.line_delimiter):
        if line.startswith(prefixes.category):
            header = parse_category_code(line, prefixes.category)
        elif line.startswith(prefixes.time):
            parsed_time = parse_time_line(
                line, prefixes.time, tokenization_cfg.time_unspecified, tokenization_cfg.delimiter
            )
            if parsed_time is None:
                continue
            hours, minutes = parsed_time
            current_time = current_time.replace(hour=hours, minute=minutes)
            if current_time > prediction_end_time:
                finished = True
                break
        elif line.startswith(prefixes.delta):
            delta_days = parse_delta_line(line, prefixes.delta)
            current_time += timedelta(days=delta_days)
            if current_time > prediction_end_time:
                finished = True
                break
        else:
            event_str = f"{header}{tokenization_cfg.delimiter}{line}"
            timestamps.append(current_time)
            codes.append(event_str)
            if any(event_str.startswith(end_code) for end_code in end_codes):
                found, finished = True, True
                break

    return timestamps, codes, found, finished


def batch_inference(
    ehr_texts: list[str],
    prediction_times: list[datetime],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenization_cfg: DictConfig,
    end_time_delta: timedelta | None,
    end_codes: list[str] | None,
    generation_config: DictConfig,
    stopping_criteria: StoppingCriteriaList | None,
) -> dict[str, list]:
    """Run inference on a batch of EHR texts by
    - Tokenizing the input texts with appropriate padding and truncation
    - Generating continuations using the model's generate method
    - Post-processing the generated tokens to extract predicted events, compute probabilities,
    and check format adherence
    - Measuring latency for the batch and computing average latency per sample
    """
    start_time = time.perf_counter()

    ## Add line delimiter to end of each input to prompt model to generate new event
    ehr_texts = [text + tokenization_cfg.line_delimiter for text in ehr_texts]

    inputs = tokenizer(
        ehr_texts,
        truncation=True,
        max_length=tokenizer.model_max_length - generation_config.max_new_tokens,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass to get logits for the next token (pass tensors explicitly)
    with torch.inference_mode():
        outputs = model.generate(
            stopping_criteria=stopping_criteria,
            **inputs,
            **generation_config,
        )

    # Extract generated tokens from outputs, reshape to separate sequences, and convert to list for post-processing
    start_idx = int(inputs["input_ids"].shape[1])
    batch_new_tokens = outputs[:, start_idx:]
    batch_new_tokens = batch_new_tokens.view(len(ehr_texts), generation_config.get("num_return_sequences", 1), -1)
    batch_new_tokens = batch_new_tokens.cpu().tolist()

    # Post-process outputs to extract generated tokens, compute probabilities, and check format adherence
    results: dict[str, list[Any]] = defaultdict(list)
    for ehr_text, ehr_new_tokens, prediction_time in zip(ehr_texts, batch_new_tokens, prediction_times):
        ehr_results = defaultdict(list)
        prediction_header = extract_last_category_content(
            ehr_text, tokenization_cfg.prefixes.category, tokenization_cfg.delimiter
        )
        for new_tokens in ehr_new_tokens:
            completion = tokenization_cfg.delimiter.join(tokenizer.convert_ids_to_tokens(new_tokens))
            timestamps, codes, found, finished = parse_completion(
                completion, prediction_time, prediction_header, end_time_delta, end_codes, tokenization_cfg
            )

            ehr_results["completion"].append(completion)
            ehr_results["timestamps"].append(timestamps)
            ehr_results["codes"].append(codes)
            ehr_results["probability"].append(int(found))
            ehr_results["format_adherence"].append(int(finished))

        for key, values in ehr_results.items():
            results[key].append(values)

        results["mean_probability"].append(sum(ehr_results["probability"]) / len(ehr_results["probability"]))
        results["mean_format_adherence"].append(
            sum(ehr_results["format_adherence"]) / len(ehr_results["format_adherence"])
        )

    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / len(ehr_texts)
    results["sample_latency"] = [sample_latency] * len(ehr_texts)  # noqa: mypy

    return results


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_generative_rollout",
)
def main(cfg: DictConfig) -> None:
    """
    Main function for structured generative rollout inference. This function:
    1. Sets up logging and output directories
    2. Loads the model and tokenizer once for efficiency
    3. Iterates through each task in the specified dataset split, loading the data and running batched inference to
    generate predictions
    4. Post-processes the generated outputs to extract predicted events, compute probabilities, and check format
    adherence based on defined end conditions
    5. Saves the results for each task to disk in a structured format for later evaluation

    Configurations for model loading, tokenization, generation, and end conditions are provided via Hydra config"""
    log.info("Starting Structured Generative Rollout Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir)
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=cast(Type[PreTrainedModel], AutoModelForCausalLM),
        model_config=cfg.model,
        eval_mode=True,
        padding_side="left",
    )

    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        shards = list((task_dir / cfg.split).glob("*.parquet"))
        if not shards:
            log.warning(f"No shard files found for task {task_name} in split {cfg.split}. Skipping.")
            continue
        dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
        log.info(f"Loaded {len(dataset)} examples from task directory.")

        # Get task end conditions for parsing completions and determining prediction windows
        end_conditions = cfg.end_conditions.get(task_name, {})
        end_codes = end_conditions.get("codes", None)
        if not end_codes:
            log.warning(f"No end codes defined for task {task_name}.")

        end_time = end_conditions.get("time", None)
        if end_time is None:
            end_time_delta = None
            log.warning(f"No prediction window defined for task {task_name}.")
        else:
            end_time_delta = timedelta(
                days=end_time.get("days", 0), hours=end_time.get("hours", 0), minutes=end_time.get("minutes", 0)
            )

        stopping_criteria = stopping_criteria = (
            StoppingCriteriaList(
                [
                    WordLevelStopCriteria(
                        stop_strings=end_codes,
                        tokenizer=tokenizer,
                    )
                ]
            )
            if end_codes
            else None
        )

        # Sort to minimize padding within batches
        dataset = dataset.map(
            lambda ehr_text: {"length": len(ehr_text)},
            input_columns=["ehr_text"],
            num_proc=cfg.num_workers,
        )
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns("length")

        try:
            # Run batched inference
            dataset = dataset.map(
                lambda ehr_texts, prediction_times: batch_inference(
                    ehr_texts,
                    prediction_times=prediction_times,
                    model=model,
                    tokenizer=tokenizer,
                    tokenization_cfg=cfg.tokenization,
                    generation_config=cfg.generation,
                    end_codes=end_codes,
                    end_time_delta=end_time_delta,
                    stopping_criteria=stopping_criteria,
                ),
                input_columns=["ehr_text", "prediction_time"],
                batched=True,
                batch_size=cfg.batch_size,
            )

            # Save outputs
            out_file_path = out_dir / cfg.split / task_dir.name
            dataset.save_to_disk(out_file_path)
            log.info(f"Saved results to {out_file_path}")

        except Exception as e:
            log.error(f"Error processing task {task_dir.name}: {e}")
            raise

    log.info("Inference Complete!")
