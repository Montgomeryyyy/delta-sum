"""Inference script for structured generative rollout."""

import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteriaList

from mimic_struct_predict.models.structured.stopping_criteria import TimeDeltaStoppingCriteria, WordLevelStopCriteria
from mimic_struct_predict.models.utils import (
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


def extract_last_category_content(text: str, category_prefix: str, delimiter: str) -> str:
    """Extracts the content of the last category header in the text.
    Expected format is '[CATEGORY PREFIX][content][DELIMITER]'."""
    # 1. Search backwards from the end of the text for the last prefix
    start_idx = text.rfind(category_prefix)
    if start_idx != -1:
        # Move our index forward past the prefix itself
        content_start = start_idx + len(category_prefix)
        # 2. Search forwards from that exact spot to find the closing delimiter
        end_idx = text.find(delimiter, content_start)

        if end_idx != -1:
            return text[content_start:end_idx]

    log.warning("No category header found in the text. This may indicate a formatting issue.")
    return "Unknown"


def parse_category_code(line: str, category_prefix: str) -> str:
    """Extracts the category code from a header line. Expected format is '[CATEGORY PREFIX][code]'."""
    return line.removeprefix(category_prefix)


def parse_time_line(line: str, time_prefix: str, time_unspecified: str, delimiter: str) -> tuple[int, int] | None:
    """Extracts hours and minutes from a time line.
    Expected format is '[TIME PREFIX][HH]h[delimiter][MM]m' or '[TIME PREFIX]Unspecified'.
    Returns a tuple of (hours, minutes) or None if the format is invalid.
    """
    time_content = line.removeprefix(time_prefix)

    if time_content == time_unspecified:
        return 23, 59  # Default to end of day if time is unspecified

    # Matches patterns like "12h//30m", "12 h // 30 m", or "09h//05m" safely
    escaped_delim = re.escape(delimiter)
    pattern = rf"^(\d+)\s*h\s*{escaped_delim}\s*(\d+)\s*m$"
    match = re.match(pattern, time_content, re.IGNORECASE)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        # Strict bound checking
        if 0 <= hours <= 23 and 0 <= minutes <= 59:
            return hours, minutes
        else:
            log.warning(f"Time out of bounds (HH:MM): {line}. Keeping current time unchanged.")
            return None

    log.warning(f"Invalid time format: {line}. Keeping current time unchanged.")
    return None


def parse_delta_line(line: str, delta_prefix: str) -> int:
    """Extracts the number of days from a delta line. Expected format is '[DELTA PREFIX][+][N] days'."""
    delta_content = line.removeprefix(delta_prefix)
    try:
        days_str = delta_content.split(" ")[0]
        delta_days = int(days_str.lstrip("+"))
        return delta_days
    except (ValueError, IndexError):
        log.warning(f"Invalid delta format: {line}. No time delta will be applied.")
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
    ).to(model.device)

    stopping_criteria = []
    if end_codes is not None:
        stopping_criteria.append(WordLevelStopCriteria(end_codes, tokenizer))
    if end_time_delta is not None:
        stopping_criteria.append(
            TimeDeltaStoppingCriteria(tokenizer, end_time_delta.days, inputs["input_ids"].shape[1])
        )

    # Forward pass to get logits for the next token (pass tensors explicitly)
    with torch.inference_mode():
        outputs = model.generate(
            stopping_criteria=StoppingCriteriaList(stopping_criteria),
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
            # Assuming use of word-level tokenizer where each token corresponds to a complete event token,
            # so we can directly convert tokens to strings and join them with the delimiter for parsing.
            tokens = tokenizer.convert_ids_to_tokens(new_tokens, skip_special_tokens=True)
            completion = tokenization_cfg.delimiter.join(
                tokenizer.convert_ids_to_tokens(new_tokens, skip_special_tokens=True)
            )
            timestamps, codes, found, finished = parse_completion(
                completion, prediction_time, prediction_header, end_time_delta, end_codes, tokenization_cfg
            )

            ehr_results["tokens"].append(tokens)
            ehr_results["completion"].append(completion)
            ehr_results["timestamps"].append(timestamps)
            ehr_results["codes"].append(codes)
            ehr_results["sample_probabilities"].append(int(found))
            ehr_results["sample_format_adherences"].append(int(finished))

        for key, values in ehr_results.items():
            results[key].append(values)

        results["probability"].append(
            sum(ehr_results["sample_probabilities"]) / len(ehr_results["sample_probabilities"])
        )
        results["format_adherence"].append(
            sum(ehr_results["sample_format_adherences"]) / len(ehr_results["sample_format_adherences"])
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
        model_class=AutoModelForCausalLM,
        model_config=cfg.model,
        eval_mode=True,
        padding_side="left",
    )

    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name == ".logs":
            continue

        task_name = task_dir.name
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        shards = (task_dir / cfg.split).glob("*.parquet")
        if not shards:
            log.warning(f"No shard files found for task {task_name} in split {cfg.split}. Skipping.")
            continue
        dataset = datasets.Dataset.from_parquet([str(p) for p in shards])
        log.info(f"Loaded {len(dataset)} examples from task directory.")

        if not dataset.features["prediction_time"].dtype.startswith("timestamp"):
            raise ValueError(
                "Dataset 'prediction_time' column is in timestamp format."
                "Please convert it to ISO string or datetime before inference."
            )

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

        # Sort to minimize padding within batches
        dataset = dataset.map(
            lambda ehr_text: {"length": len(ehr_text)},
            input_columns=["ehr_text"],
            num_proc=cfg.num_workers,
        )
        dataset = dataset.sort("length", reverse=True)
        dataset = dataset.remove_columns("length")

        writer = None
        out_file_path = out_dir / cfg.split / task_dir.name / "results.parquet"
        out_file_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Running inference and streaming to {out_file_path}")
        try:
            for i in tqdm(range(0, len(dataset), cfg.batch_size)):
                batch = dataset[i : i + cfg.batch_size]
                results = batch_inference(
                    ehr_texts=batch["ehr_text"],
                    prediction_times=batch["prediction_time"],
                    model=model,
                    tokenizer=tokenizer,
                    tokenization_cfg=cfg.tokenization,
                    generation_config=cfg.generation,
                    end_codes=end_codes,
                    end_time_delta=end_time_delta,
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
