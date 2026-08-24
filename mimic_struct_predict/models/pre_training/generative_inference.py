"""HuggingFace vLLM inference datasets with dataset sharding for autoregressive rollout."""

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
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

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
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to be a dictionary, got {type(config).__name__}")

    valid_categories: set[str] = set()

    for table_name, table_config in config.items():
        if not isinstance(table_config, dict):
            continue
        for event_name, event_config in table_config.items():
            if not isinstance(event_config, dict):
                continue
            code = event_config.get("code")
            if code is None:
                continue
            if isinstance(code, list) and len(code) > 0:
                category = code[0]
                if category:
                    valid_categories.add(str(category))
            elif isinstance(code, str) and code:
                valid_categories.add(code)

    return valid_categories


def batch_ids_to_string(
    batch_token_ids: list[list[int]],
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizerBase,
    join_with_delimiter: bool,
    delimiter: str,
) -> list[str]:
    if join_with_delimiter:
        # HACK: Batch decode always joins using spaces, so do slightly slower way to allow for delimiter
        # join for word-level tokenizers e.g. structured
        return [
            delimiter.join(tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True))
            for token_ids in batch_token_ids
        ]
    else:
        output: list[str] = tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)
        return output


def hours_and_mins_to_days(hrs: int, mins: int) -> float:
    return hrs / 24.0 + mins / (24.0 * 60.0)


def get_last_time_offset_and_category(
    in_text: str,
    code_splitter: str,
    category_prefix: str,
    time_prefix: str,
    time_pattern: re.Pattern,
) -> dict[str, float | str]:
    last_time_offset = None
    last_category = None

    lines = in_text.split(code_splitter)
    for line in reversed(lines):
        line = line.strip()

        if last_category is None and line.startswith(category_prefix):
            last_category = line

        # Date comes after time so do not need to parse date prefixes in reverse
        if last_time_offset is None and line.startswith(time_prefix):
            if line == f"{time_prefix}Unspecified":
                last_time_offset = hours_and_mins_to_days(TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS)
            elif (
                (time_match := time_pattern.search(line))
                and 0 <= (hours := int(time_match.group("hours"))) <= 23
                and 0 <= (minutes := int(time_match.group("minutes"))) <= 59
            ):
                last_time_offset = hours_and_mins_to_days(hours, minutes)

        if last_category is not None and last_time_offset is not None:
            break

    if last_category is None:
        log.warning(f"No category found in:\n{in_text}")
        last_category = ""
    if last_time_offset is None:
        log.warning(f"No time offset found in:\n{in_text}")
        last_time_offset = 0.0

    return {"initial_time": last_time_offset, "initial_category": last_category}


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
        batch_input_ids = batch["input_ids"]
        if max_input_length is not None:
            truncated_ids: list[list[int]] = []
            for ids in batch_input_ids:
                if len(ids) > max_input_length:
                    if keep_first_token:
                        if max_input_length <= 1:
                            truncated_ids.append(ids[:max_input_length])
                        else:
                            truncated_ids.append([ids[0]] + ids[-(max_input_length - 1) :])
                    else:
                        truncated_ids.append(ids[-max_input_length:])
                else:
                    truncated_ids.append(ids)
        else:
            truncated_ids = batch_input_ids

        decoded_inputs = batch_ids_to_string(truncated_ids, tokenizer, join_with_delimiter, delimiter)

        initial_times: list[float] = []
        initial_categories: list[str] = []
        for text in decoded_inputs:
            res = get_last_time_offset_and_category(text, code_splitter, category_prefix, time_prefix, time_pattern)
            initial_times.append(float(res["initial_time"]))
            initial_categories.append(str(res["initial_category"]))

        return {
            "input_ids": truncated_ids,
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
    """Returns parsed outcomes from the LLM output as a dictionary."""

    current_time = initial_time_offset
    current_category = initial_category

    outcome_predicted = None
    finish_reason = None
    num_events = 0

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
                    (time_match := time_pattern.search(line))
                    and 0 <= (hours := int(time_match.group("hours"))) <= 23
                    and 0 <= (minutes := int(time_match.group("minutes"))) <= 59
                ):
                    next_time = math.floor(current_time) + hours_and_mins_to_days(hours, minutes)
                else:
                    finish_reason = "Time Error"
            elif line.startswith(date_prefix):
                if (date_match := date_pattern.search(line)) and 1 <= (
                    days := int(date_match.group("days"))
                ) <= max_day_gap:
                    next_time = math.floor(current_time) + days
                else:
                    finish_reason = "Date Error"
            else:
                num_events += 1
                event_str = f"{current_category}{delimiter}{line}"
                if target_event_regex.search(event_str):
                    outcome_predicted = True
                    finish_reason = "Target Event Found"
                elif end_event_regex is not None and end_event_regex.search(event_str):
                    outcome_predicted = False
                    finish_reason = "End Event Found"

            if finish_reason is None and next_time != current_time:
                if next_time < current_time:
                    finish_reason = "Time Error"
                elif end_time_days is not None and (next_time - initial_time_offset) > end_time_days:
                    outcome_predicted = False
                    finish_reason = "End Time Reached"

            current_time = next_time

            if finish_reason is not None:
                # If a formatting error occurred on the final line, it is likely due to truncation
                if finish_reason not in SUCCESS_FINISH_REASONS and line_idx == len(lines) - 1:
                    finish_reason = "Max New Tokens Reached"
                break

        if finish_reason is None:
            finish_reason = "Max New Tokens Reached"

    except Exception as e:
        log.error(f"Error parsing outcome: {e}", exc_info=True)
        finish_reason = "Formatting Error"

    format_adherence = finish_reason in SUCCESS_FINISH_REASONS
    time_covered = current_time - initial_time_offset

    return {
        "outcome_predicted": outcome_predicted,
        "format_adherence": format_adherence,
        "finish_reason": finish_reason,
        "num_events": num_events,
        "time_covered": time_covered,
    }


@hydra.main(
    version_base=None,
    config_path="config/generative_inference",
)
def main(cfg: DictConfig) -> None:
    """Run LLM inference and process outputs into Parquet chunks."""

    log.info("Starting LLM Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir)
    out_dir = Path(hydra_cfg.run.dir)
    inference_dir = out_dir
    inference_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Inference directory: {inference_dir}")

    dataset = datasets.load_from_disk(str(dataset_dir))
    if cfg.get("max_eval_samples") is not None:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        log.info(f"Running distributed: Rank {rank}/{world_size}. Sharding dataset...")
        dataset = dataset.shard(num_shards=world_size, index=rank)
    log.info(f"Loaded {len(dataset)} examples from data directory (after sharding).")

    ### Preprocessing ###
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

    task_prevalence_path = Path(hydra.utils.to_absolute_path(str(cfg.task_prevalence_path)))
    if not task_prevalence_path.exists():
        raise FileNotFoundError(f"Task prevalence file not found: {task_prevalence_path}")
    task_prevalences = json.loads(task_prevalence_path.read_text(encoding="utf-8"))
    if cfg.task not in task_prevalences:
        raise KeyError(f"Task '{cfg.task}' not found in task prevalence file: {task_prevalence_path}")
    task_prevalence = float(task_prevalences[cfg.task])
    log.info(f"Loaded task prevalence for '{cfg.task}': {task_prevalence}")

    event_config_path = hydra.utils.to_absolute_path(str(cfg.event_extraction.event_config_path))
    valid_categories = get_valid_categories(event_config_path)
    log.info(f"Loaded {len(valid_categories)} valid categories from {event_config_path}")

    escaped_delimiter = re.escape(cfg.event_extraction.delimiter)
    escaped_date_prefix = re.escape(cfg.event_extraction.date_prefix)
    escaped_time_prefix = re.escape(cfg.event_extraction.time_prefix)

    date_pattern = re.compile(rf"^{escaped_date_prefix}\+(?P<days>\d+) days$")
    time_pattern = re.compile(rf"^{escaped_time_prefix}(?P<hours>\d+)h{escaped_delimiter}(?P<minutes>\d+)m$")

    # Get task specific regex and time conditions
    task_config = cfg.event_extraction.tasks[cfg.task]
    target_event_regex = re.compile(task_config.target_event_regex)
    end_event_str = task_config.end_event_regex
    end_event_regex = re.compile(end_event_str) if end_event_str is not None else None
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

    ### Load model ###
    log.info(f"Loading model ({cfg.model.model})...")
    llm = LLM(**cfg.model)

    sampling_params_dict = OmegaConf.to_container(cfg.sampling_params, resolve=True)
    set_max_token_id = sampling_params_dict.pop("set_max_token_id", False)
    if set_max_token_id:
        vocab_size = len(tokenizer)
        log.info(f"Setting max token id to tokenizer vocab size of {vocab_size}")
        sampling_params_dict["allowed_token_ids"] = list(range(0, vocab_size))

    sampling_params = SamplingParams(**sampling_params_dict)
    log.info(sampling_params)

    # vLLM batch inference in chunks
    total_chunks = math.ceil(len(dataset) / cfg.chunk_size)
    for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
        chunk_index = i // cfg.chunk_size
        log.info(f"Processing Chunk {chunk_index + 1} / {total_chunks}")
        chunk_file_path = inference_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

        chunk = dict(dataset[i : i + cfg.chunk_size])

        prompts = [{"prompt_token_ids": ids} for ids in chunk.pop("input_ids")]

        outputs = llm.generate(prompts, sampling_params=sampling_params)
        flat_output_ids = [output.token_ids for request_output in outputs for output in request_output.outputs]

        flat_decoded = batch_ids_to_string(
            flat_output_ids, tokenizer, cfg.tokenizer.join_with_delimiter, cfg.tokenizer.delimiter
        )

        n = cfg.sampling_params.n
        chunk["decoded_outputs"] = [
            flat_decoded[start_idx : start_idx + n] for start_idx in range(0, len(flat_decoded), n)
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

            # Handle error cases
            outcomes = sample_rollout_results["outcome_predicted"]
            num_outcomes = len(outcomes)

            prob_0 = sum(1.0 if o is True else 0.0 for o in outcomes) / num_outcomes

            prob_0_5 = sum(1.0 if o is True else (0.0 if o is False else 0.5) for o in outcomes) / num_outcomes

            prob_prevalence = (
                sum(1.0 if o is True else (0.0 if o is False else task_prevalence) for o in outcomes) / num_outcomes
                if num_outcomes > 0
                else task_prevalence
            )

            # Drop trajectory (if all trajectories dropped default to prevalence)
            valid_outcomes = [1.0 if o is True else 0.0 for o in outcomes if o is not None]
            prob_drop_trajectory = sum(valid_outcomes) / len(valid_outcomes) if valid_outcomes else task_prevalence

            sample_rollout_results["probability_0"].append(prob_0)
            sample_rollout_results["probability_0_5"].append(prob_0_5)
            sample_rollout_results["probability_prevalence"].append(prob_prevalence)
            sample_rollout_results["probability_drop_trajectory"].append(prob_drop_trajectory)

            for k, v in sample_rollout_results.items():
                chunk_results[k].append(v)

        chunk.update(chunk_results)

        log.info(f"Saving to {chunk_file_path}")
        table = pa.Table.from_pydict(chunk)
        pq.write_table(table, chunk_file_path)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
