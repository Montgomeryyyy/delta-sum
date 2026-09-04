"""HuggingFace Transformers inference with dataset sharding for autoregressive rollout."""

import json
import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    set_seed,
)

log = logging.getLogger(__name__)

TIME_UNSPECIFIED_HRS = 23
TIME_UNSPECIFIED_MINS = 59
SUCCESS_FINISH_REASONS = {"Target Event Found", "End Event Found", "End Time Reached"}


def get_valid_categories(event_config_path: Path | str) -> set[str]:
    """Extract valid category prefixes from an event configuration YAML file."""
    path = Path(event_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Event config file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    valid_categories: set[str] = set()

    for table_config in config.values():
        if not isinstance(table_config, dict):
            continue
        for event_config in table_config.values():
            if not isinstance(event_config, dict):
                continue
            code = event_config.get("code")
            if isinstance(code, list) and code:
                valid_categories.add(str(code[0]))
            elif isinstance(code, str) and code:
                valid_categories.add(code)

    return valid_categories


def batch_ids_to_string(
    batch_token_ids: list[list[int]] | torch.Tensor,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
    join_with_delimiter: bool,
    delimiter: str,
) -> list[str]:
    """Decode batch token IDs to strings, optionally joining via custom delimiter."""
    if join_with_delimiter:
        # HACK: Batch decode always joins using spaces, so do slightly slower way to allow for delimiter
        # join for word-level tokenizers e.g. structured
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.tolist()
        return [
            delimiter + delimiter.join(tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True))
            for token_ids in batch_token_ids
        ]
    decoded = tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)
    return [str(s) for s in decoded]


def hours_and_mins_to_days(hrs: int, mins: int) -> float:
    return hrs / 24.0 + mins / (24.0 * 60.0)


def get_last_time_offset_and_category(
    in_text: str,
    code_splitter: str,
    category_prefix: str,
    time_prefix: str,
    time_pattern: re.Pattern,
    delimiter: str,
) -> dict[str, float | str]:
    """Extract the last time offset and category from prompt text in reverse."""
    last_time_offset = None
    last_category = None

    if in_text.startswith(delimiter):
        in_text = in_text[len(delimiter) :]

    for line in reversed(in_text.split(code_splitter)):
        line = line.strip()
        if delimiter and line.startswith(delimiter):
            line = line[len(delimiter) :].strip()
        if last_category is None and line.startswith(category_prefix):
            last_category = line

        # Date comes after time so do not need to parse date prefixes in reverse
        if last_time_offset is None and line.startswith(time_prefix):
            if line == f"{time_prefix}Unspecified":
                last_time_offset = hours_and_mins_to_days(TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS)
            elif (
                (time_match := time_pattern.match(line))
                and 0 <= (hours := int(time_match.group("hours"))) <= 23
                and 0 <= (minutes := int(time_match.group("minutes"))) <= 59
            ):
                last_time_offset = hours_and_mins_to_days(hours, minutes)

        if last_category is not None and last_time_offset is not None:
            break

    if last_category is None:
        log.warning(f"No category found in prompt text:\n{in_text}")
    if last_time_offset is None:
        log.warning(f"No time offset found in prompt text:\n{in_text}")

    return {
        "initial_time": last_time_offset if last_time_offset is not None else 0.0,
        "initial_category": last_category or "",
    }


def truncate_ids(ids: list[int], max_len: int | None, keep_first_token: bool) -> list[int]:
    """Truncate token IDs from the left, optionally preserving the first token."""
    if max_len is None or len(ids) <= max_len:
        return ids
    if keep_first_token and max_len > 1:
        return [ids[0]] + ids[-(max_len - 1) :]
    return ids[-max_len:]


def get_preprocess_batch_function(
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
    join_with_delimiter: bool,
    delimiter: str,
    max_input_length: int | None,
    keep_first_token: bool,
    code_splitter: str,
    category_prefix: str,
    time_prefix: str,
    time_pattern: re.Pattern,
) -> Callable[[dict[str, list[Any]]], dict[str, Any]]:
    def preprocess_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        """Truncate input IDs, decode batch to text, and extract initial time and category."""
        truncated_ids = [truncate_ids(ids, max_input_length, keep_first_token) for ids in batch["input_ids"]]
        decoded_inputs = batch_ids_to_string(truncated_ids, tokenizer, join_with_delimiter, delimiter)

        initial_times: list[float] = []
        initial_categories: list[str] = []
        for text in decoded_inputs:
            res = get_last_time_offset_and_category(
                text, code_splitter, category_prefix, time_prefix, time_pattern, delimiter
            )
            initial_times.append(float(res["initial_time"]))
            initial_categories.append(str(res["initial_category"]))

        return {
            "input_ids": truncated_ids,
            "len_input_ids": [len(ids) for ids in truncated_ids],
            "decoded_input": decoded_inputs,
            "initial_time": initial_times,
            "initial_category": initial_categories,
        }

    return preprocess_batch


def detect_outcome(
    text: str,
    code_splitter: str,
    delimiter: str,
    category_prefix: str,
    time_prefix: str,
    date_prefix: str,
    date_pattern: re.Pattern,
    time_pattern: re.Pattern,
    target_event_regex: re.Pattern,
    end_event_regex: re.Pattern | None,
    end_time_days: float | None,
    initial_time_offset: float,
    initial_category: str,
    valid_categories: set[str],
    max_day_gap: int,
) -> dict[str, Any]:
    """Parse rollout outcome, format adherence, and timeline from LLM generated text."""
    current_time = initial_time_offset
    current_category = initial_category

    outcome_predicted = None
    finish_reason = None
    num_events = 0
    line_idx = 0

    lines = [line.strip() for line in text.split(code_splitter) if line.strip()]

    try:
        for line_idx, line in enumerate(lines):
            next_time = current_time

            if line.startswith(category_prefix):
                category = line.removeprefix(category_prefix)
                if category in valid_categories:
                    current_category = line
                else:
                    finish_reason = "Category Error"
            elif line.startswith(time_prefix):
                if line == f"{time_prefix}Unspecified":
                    next_time = math.floor(current_time) + hours_and_mins_to_days(
                        TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS
                    )
                elif (
                    (time_match := time_pattern.match(line))
                    and 0 <= (hours := int(time_match.group("hours"))) <= 23
                    and 0 <= (minutes := int(time_match.group("minutes"))) <= 59
                ):
                    next_time = math.floor(current_time) + hours_and_mins_to_days(hours, minutes)
                else:
                    finish_reason = "Time Error"
            elif line.startswith(date_prefix):
                if (date_match := date_pattern.match(line)) and 1 <= (
                    days := int(date_match.group("days"))
                ) <= max_day_gap:
                    next_time = math.floor(current_time) + days
                else:
                    finish_reason = "Date Error"
            else:
                num_events += 1
                event_str = f"{current_category}{delimiter}{line}"
                if target_event_regex.match(event_str):
                    outcome_predicted = True
                    finish_reason = "Target Event Found"
                elif end_event_regex is not None and end_event_regex.match(event_str):
                    outcome_predicted = False
                    finish_reason = "End Event Found"

            if finish_reason is None and next_time != current_time:
                if next_time < current_time:
                    finish_reason = "Time Backward Error"
                elif end_time_days is not None and (next_time - initial_time_offset) > end_time_days:
                    outcome_predicted = False
                    finish_reason = "End Time Reached"

            current_time = next_time

            if finish_reason is not None:
                # If formatting error occurred on the final line, likely truncation due to max_new_tokens
                if finish_reason not in SUCCESS_FINISH_REASONS and line_idx == len(lines) - 1:
                    finish_reason = "Max New Tokens Reached"
                break

        if finish_reason is None:
            finish_reason = "Max New Tokens Reached"

    except Exception as e:
        log.error(f"Error parsing outcome: {e}", exc_info=True)
        finish_reason = "Formatting Error"
        line_idx = len(lines) - 1 if lines else 0

    format_adherence = finish_reason in SUCCESS_FINISH_REASONS
    time_covered = current_time - initial_time_offset
    error_line = lines[line_idx] if (not format_adherence and lines and 0 <= line_idx < len(lines)) else ""

    return {
        "outcome_predicted": outcome_predicted,
        "format_adherence": format_adherence,
        "finish_reason": finish_reason,
        "num_events": num_events,
        "num_lines": line_idx + 1 if lines else 0,
        "time_covered": time_covered,
        "error_line": error_line,
    }


def compute_rollout_probabilities(outcomes: list[bool | None], task_prevalence: float) -> dict[str, float]:
    """Compute binary outcome probability estimates across sampled rollouts."""
    num_outcomes = len(outcomes)
    if num_outcomes == 0:
        return {
            "probability_0": 0.0,
            "probability_0_5": 0.5,
            "probability_prevalence": task_prevalence,
            "probability_drop_trajectory": task_prevalence,
        }

    prob_0 = sum(1.0 for o in outcomes if o is True) / num_outcomes
    prob_0_5 = sum(1.0 if o is True else (0.0 if o is False else 0.5) for o in outcomes) / num_outcomes
    prob_prevalence = (
        sum(1.0 if o is True else (0.0 if o is False else task_prevalence) for o in outcomes) / num_outcomes
    )

    valid_outcomes = [1.0 if o is True else 0.0 for o in outcomes if o is not None]
    prob_drop_trajectory = sum(valid_outcomes) / len(valid_outcomes) if valid_outcomes else task_prevalence

    return {
        "probability_0": prob_0,
        "probability_0_5": prob_0_5,
        "probability_prevalence": prob_prevalence,
        "probability_drop_trajectory": prob_drop_trajectory,
    }


@hydra.main(
    version_base=None,
    config_path="config/generative_inference_hf",
)
def main(cfg: DictConfig) -> None:
    """Run HuggingFace LLM inference and process outputs into Parquet chunks."""
    log.info("Starting HuggingFace LLM Inference (v2)")

    out_dir = Path(HydraConfig.get().run.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Inference directory: {out_dir}")

    if cfg.get("seed") is not None:
        set_seed(cfg.seed)

    # Load dataset with optional subsampling and multi-GPU sharding
    dataset = datasets.load_from_disk(str(cfg.dataset_dir))
    if cfg.get("max_eval_samples") is not None:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        log.info(f"Running distributed: Rank {rank}/{world_size} (Local Rank {local_rank}). Sharding dataset...")
        dataset = dataset.shard(num_shards=world_size, index=rank)
    log.info(f"Loaded {len(dataset)} examples after sharding.")

    # Tokenizer setup
    model_name_or_path = cfg.model.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Task prevalence
    task_prevalence_path = Path(hydra.utils.to_absolute_path(str(cfg.task_prevalence_path)))
    if not task_prevalence_path.exists():
        raise FileNotFoundError(f"Task prevalence file not found: {task_prevalence_path}")
    task_prevalences = json.loads(task_prevalence_path.read_text(encoding="utf-8"))
    if cfg.task not in task_prevalences:
        raise KeyError(f"Task '{cfg.task}' not found in task prevalence file: {task_prevalence_path}")
    task_prevalence = float(task_prevalences[cfg.task])
    log.info(f"Loaded task prevalence for '{cfg.task}': {task_prevalence}")

    # Event extraction regex patterns
    event_config_path = hydra.utils.to_absolute_path(str(cfg.event_extraction.event_config_path))
    valid_categories = get_valid_categories(event_config_path)
    log.info(f"Loaded {len(valid_categories)} valid categories from {event_config_path}")

    escaped_delimiter = re.escape(cfg.event_extraction.delimiter)
    date_pattern = re.compile(rf"^{re.escape(cfg.event_extraction.date_prefix)}\+(?P<days>\d+) days$")
    time_pattern = re.compile(
        rf"^{re.escape(cfg.event_extraction.time_prefix)}(?P<hours>\d+)h{escaped_delimiter}(?P<minutes>\d+)m$"
    )

    task_config = cfg.event_extraction.tasks[cfg.task]
    target_event_regex = re.compile(task_config.target_event_regex)
    end_event_regex = re.compile(task_config.end_event_regex) if task_config.end_event_regex is not None else None
    end_time_days = task_config.end_time_days
    if end_event_regex is None and end_time_days is None:
        raise ValueError("end_event_regex and end_time_days cannot both be None")

    log.info(f"Target Regex: {target_event_regex}, End Regex: {end_event_regex}, End Time: {end_time_days}")

    # Truncate, decode, and extract initial state in a single preprocessing pass
    preprocess_batch = get_preprocess_batch_function(
        tokenizer=tokenizer,
        join_with_delimiter=cfg.tokenizer.join_with_delimiter,
        delimiter=cfg.tokenizer.delimiter,
        max_input_length=cfg.tokenizer.max_input_length,
        keep_first_token=cfg.tokenizer.keep_first_token,
        code_splitter=cfg.event_extraction.code_splitter,
        category_prefix=cfg.event_extraction.category_prefix,
        time_prefix=cfg.event_extraction.time_prefix,
        time_pattern=time_pattern,
    )
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        num_proc=cfg.num_workers,
        desc="Preprocessing and decoding batch inputs",
    )
    log.info("Sorting dataset by input_ids length...")
    dataset = dataset.sort("len_input_ids", reverse=True).remove_columns("len_input_ids")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    # Load model with native HF arguments
    log.info(f"Loading model ({model_name_or_path}) onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(**OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    model.eval()

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Configure generation
    gen_dict = OmegaConf.to_container(cfg.generation_config, resolve=True)
    gen_config = GenerationConfig(pad_token_id=tokenizer.pad_token_id, **gen_dict)

    if cfg.get("suppress_extra_vocab", False):
        vocab_size = len(tokenizer)
        model_vocab_size = getattr(model.config, "vocab_size", vocab_size)
        if model_vocab_size > vocab_size:
            log.info(f"Suppressing extra vocab token IDs from {vocab_size} to {model_vocab_size}")
            gen_config.suppress_tokens = list(range(vocab_size, model_vocab_size))

    batch_size = int(cfg.batch_size)
    num_return_sequences = int(gen_config.num_return_sequences)
    log.info(f"GenerationConfig: {gen_config}, batch_size: {batch_size}")

    total_chunks = math.ceil(len(dataset) / cfg.chunk_size)
    for i in tqdm(range(0, len(dataset), cfg.chunk_size), total=total_chunks, desc="Processing Chunks"):
        chunk_index = i // cfg.chunk_size
        log.info(f"Processing Chunk {chunk_index + 1} / {total_chunks}")
        chunk_file_path = out_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

        chunk = dict(dataset[i : i + cfg.chunk_size])
        chunk_input_ids: list[list[int]] = chunk.pop("input_ids")

        flat_decoded: list[str] = []
        for b_start in tqdm(range(0, len(chunk_input_ids), batch_size), desc="Generating"):
            batch_ids = chunk_input_ids[b_start : b_start + batch_size]
            encoded_inputs = tokenizer.pad(
                {"input_ids": batch_ids},
                padding=True,
                pad_to_multiple_of=cfg.pad_to_multiple_of,
                return_tensors="pt",
            ).to(model.device)

            input_seq_len = encoded_inputs["input_ids"].shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    **encoded_inputs,
                    generation_config=gen_config,
                )
            generated_tokens = outputs[:, input_seq_len:].cpu()
            batch_decoded = batch_ids_to_string(
                generated_tokens, tokenizer, cfg.tokenizer.join_with_delimiter, cfg.tokenizer.delimiter
            )
            flat_decoded.extend(batch_decoded)

        chunk["decoded_outputs"] = [
            flat_decoded[start : start + num_return_sequences]
            for start in range(0, len(flat_decoded), num_return_sequences)
        ]

        chunk_results: defaultdict[str, list[Any]] = defaultdict(list)
        for sample_outputs, initial_time, initial_category in zip(
            chunk["decoded_outputs"], chunk["initial_time"], chunk["initial_category"]
        ):
            sample_rollout_results: defaultdict[str, list[Any]] = defaultdict(list)
            for out_text in sample_outputs:
                result = detect_outcome(
                    out_text,
                    code_splitter=cfg.event_extraction.code_splitter,
                    delimiter=cfg.event_extraction.delimiter,
                    category_prefix=cfg.event_extraction.category_prefix,
                    time_prefix=cfg.event_extraction.time_prefix,
                    date_prefix=cfg.event_extraction.date_prefix,
                    date_pattern=date_pattern,
                    time_pattern=time_pattern,
                    target_event_regex=target_event_regex,
                    end_event_regex=end_event_regex,
                    end_time_days=end_time_days,
                    initial_time_offset=initial_time,
                    initial_category=initial_category,
                    valid_categories=valid_categories,
                    max_day_gap=cfg.event_extraction.max_day_gap,
                )
                for k, v in result.items():
                    sample_rollout_results[k].append(v)

            # Rollout-level results (list per sample)
            for k, v in sample_rollout_results.items():
                chunk_results[k].append(v)

            # Sample-level outcome probabilities (scalar float per sample)
            probs = compute_rollout_probabilities(sample_rollout_results["outcome_predicted"], task_prevalence)
            for k, v in probs.items():
                chunk_results[k].append(v)

        chunk.update(chunk_results)

        log.info(f"Saving chunk to {chunk_file_path}")
        pq.write_table(pa.Table.from_pydict(chunk), chunk_file_path)

    log.info(f"Inference complete! Outputs saved to {out_dir}")
