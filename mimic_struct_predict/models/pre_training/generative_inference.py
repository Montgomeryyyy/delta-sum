"""HuggingFace vLLM inference datasets with dataset sharding for autoregressive rollout."""

import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)

TIME_UNSPECIFIED_HRS = 23
TIME_UNSPECIFIED_MINS = 59


def batch_ids_to_string(
    batch_token_ids: list[list[int]], tokenizer: AutoTokenizer, join_with_delimiter: bool, delimiter: str
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


def get_truncate_and_decode_batch_input_ids(
    tokenizer: PreTrainedTokenizerFast,
    join_with_delimiter: bool = False,
    delimiter: str = "",
) -> Callable[[list[list[int]], int | None, bool], dict[str, Any]]:
    def truncate_and_decode_batch_input_ids(
        batch_input_ids: list[list[int]], max_input_length: int | None, keep_first_token: bool = False
    ) -> dict[str, Any]:
        """Truncate batch input ids to max_input_length. Keep first token if keep_first_token is True."""
        if max_input_length is not None:
            input_ids = []
            for ids in batch_input_ids:
                if len(ids) > max_input_length:
                    if keep_first_token:
                        input_ids.append([ids[0]] + ids[-(max_input_length - 1) :])
                    else:
                        input_ids.append(ids[-max_input_length:])
                else:
                    input_ids.append(ids)
        else:
            input_ids = batch_input_ids

        decoded_input = batch_ids_to_string(input_ids, tokenizer, join_with_delimiter, delimiter)

        return {"input_ids": input_ids, "decoded_input": decoded_input}

    return truncate_and_decode_batch_input_ids


def hours_and_mins_to_days(hrs: int, mins: int) -> float:
    return hrs / 24.0 + mins / (24.0 * 60.0)


def get_last_time_offset_and_header(
    in_text: str, code_splitter: str, time_pattern: re.Pattern, delimiter: str = " "
) -> dict[str, float | str]:
    last_time_offset = None
    last_header = None

    lines = in_text.split(code_splitter)
    for line in reversed(lines):
        line = line.strip()

        if last_header is None and line.startswith("####"):
            last_header = line

        # Date comes after time so don't need to parse date headers in reverse
        if last_time_offset is None and line.startswith("### Time"):
            time_match = time_pattern.search(line)
            if time_match:
                last_time_offset = hours_and_mins_to_days(
                    int(time_match.group("hours")), int(time_match.group("minutes"))
                )
            elif line == f"### Time{delimiter}Unspecified":
                last_time_offset = hours_and_mins_to_days(TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS)

        if last_header is not None and last_time_offset is not None:
            break

    if last_header is None:
        log.warning(f"No header found in:\n{in_text}")
        last_header = ""
    if last_time_offset is None:
        log.warning(f"No time offset found in:\n{in_text}")
        last_time_offset = 0.0

    return {"initial_time": last_time_offset, "initial_header": last_header}


def detect_outcome(
    text: str,
    code_splitter: str,
    delimiter: str,
    date_pattern: re.Pattern,
    time_pattern: re.Pattern,
    target_event_regex: re.Pattern,
    end_event_regex: re.Pattern | None,
    end_time_days: float | None,
    initial_time_offset: float,
    initial_header: str,
) -> dict[str, Any]:
    """Returns parsed outcomes from the LLM output as a dictionary."""

    current_time = initial_time_offset
    current_header = initial_header

    outcome_predicted = False
    format_adherence = True
    finish_reason = None
    lines = text.split(code_splitter)
    line_idx = 0
    num_events = 0

    try:
        while finish_reason is None and line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line:
                line_idx += 1
                continue
            next_time = current_time

            if line.startswith("####"):
                current_header = line
            elif line.startswith("### Time"):
                time_match = time_pattern.search(line)
                if time_match:
                    hours = int(time_match.group("hours"))
                    minutes = int(time_match.group("minutes"))
                    next_time = math.floor(current_time) + hours_and_mins_to_days(hours, minutes)
                elif line == f"### Time{delimiter}Unspecified":
                    next_time = math.floor(current_time) + hours_and_mins_to_days(
                        TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS
                    )
                elif line_idx < len(lines) - 1:
                    # Don't penalize for incorrect time format due to truncation
                    log.info(f"Incorrect Time Format: {line}")
                    finish_reason = "Incorrect Time Format"
                    format_adherence = False
            elif line.startswith("## Date"):
                date_match = date_pattern.search(line)
                if date_match:
                    # Move to next N days
                    next_time = math.floor(current_time) + int(date_match.group("days"))
                elif line_idx < len(lines) - 1:
                    log.info(f"Incorrect Date Format: {line}")
                    finish_reason = "Incorrect Date Format"
                    format_adherence = False
            else:
                # It's an event see if it matches with target
                num_events += 1
                event_str = f"{current_header}{delimiter}{line}"
                if target_event_regex.search(event_str):
                    outcome_predicted = True
                    finish_reason = "Target Event Found"
                elif end_event_regex is not None and end_event_regex.search(event_str):
                    finish_reason = "End Event Found"

            if next_time != current_time:
                if next_time < current_time:
                    finish_reason = "Time went backwards"
                    format_adherence = False
                elif end_time_days is not None and (next_time - initial_time_offset) > end_time_days:
                    finish_reason = "End Time Reached"

            current_time = next_time
            line_idx += 1

        # Reached end of generation without matching regex or exceeding max time.
        if finish_reason is None:
            finish_reason = "Generation ended"
            format_adherence = False
    except Exception as e:
        log.error(f"Error parsing outcome: {e}", exc_info=True)
        finish_reason = "Formatting error"
        format_adherence = False

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
    dataset_dir = Path(cfg.dataset_dir) / cfg.split
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(hydra_cfg.run.dir)
    inference_dir = out_dir / cfg.split
    inference_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Inference directory: {inference_dir}")

    dataset = datasets.load_from_disk(str(dataset_dir)).shuffle(seed=cfg.seed)
    if cfg.get("max_eval_samples") is not None:
        dataset = dataset.select(range(cfg.max_eval_samples))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        log.info(f"Running distributed: Rank {rank}/{world_size}. Sharding dataset...")
        dataset = dataset.shard(num_shards=world_size, index=rank)
    log.info(f"Loaded {len(dataset)} examples from task directory (after sharding).")

    ### Preprocessing ###
    # 1. Truncate and decode batch input ids
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

    truncate_and_decode_batch_input_ids = get_truncate_and_decode_batch_input_ids(
        tokenizer,
        join_with_delimiter=cfg.tokenizer.join_with_delimiter,
        delimiter=cfg.tokenizer.delimiter,
    )
    dataset = dataset.map(
        lambda input_ids: truncate_and_decode_batch_input_ids(
            input_ids, cfg.tokenizer.max_input_length, cfg.tokenizer.keep_first_token
        ),
        input_columns=["input_ids"],
        batched=True,
        num_proc=cfg.num_workers,
        desc="Truncating and decoding batch input ids",
    )

    # 2. If doing event extraction get last time and header from input_ids
    event_extraction_cfg = cfg.get("event_extraction", None)
    if event_extraction_cfg is not None:
        delimiter = event_extraction_cfg["delimiter"]
        code_splitter = event_extraction_cfg["code_splitter"]
        escaped_delimiter = re.escape(delimiter)
        date_pattern = re.compile(rf"^## Date{escaped_delimiter}\+(?P<days>\d+) days$")
        time_pattern = re.compile(rf"^### Time{escaped_delimiter}(?P<hours>\d+)h{escaped_delimiter}(?P<minutes>\d+)m$")

        # Get task specific regex + time conditions
        task_config = event_extraction_cfg["tasks"][cfg.task]
        target_event_regex = re.compile(task_config["target_event_regex"])
        end_event_str = task_config.get("end_event_regex")
        end_event_regex = re.compile(end_event_str) if end_event_str is not None else None
        end_time_days = task_config.get("end_time_days")
        if end_event_regex is None and end_time_days is None:
            raise ValueError("end_event_regex and end_time_days cannot both be None")

        log.info(f"Target Regex: {target_event_regex}, End Regex: {end_event_regex}, End Time: {end_time_days}")

        # Need last time and header as init for generation
        dataset = dataset.map(
            lambda decoded_input: get_last_time_offset_and_header(
                decoded_input, code_splitter, time_pattern, delimiter
            ),
            input_columns=["decoded_input"],
            num_proc=cfg.num_workers,
            desc="Extracting last time and header from inputs",
        )

    ### Load model ###
    log.info(f"Loading model ({cfg.model.model})...")
    llm = LLM(**cfg.model)

    sampling_params_dict = OmegaConf.to_container(cfg.sampling_params, resolve=True)
    set_max_token_id = sampling_params_dict.pop("set_max_token_id", False)
    if set_max_token_id:
        log.info(f"Setting max token id to tokenizer vocab size of {tokenizer.vocab_size}")
        sampling_params_dict["allowed_token_ids"] = list(range(0, tokenizer.vocab_size))

    sampling_params = SamplingParams(**sampling_params_dict)
    log.info(sampling_params)

    # vLLM inference in chunks (so if somethings goes wrong we can restart)
    try:
        for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
            chunk_index = i // (cfg.chunk_size)
            log.info(f"Processing Chunk {chunk_index + 1}  / {len(dataset) // cfg.chunk_size}")
            chunk_file_path = inference_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

            chunk = dict(dataset[i : i + cfg.chunk_size])

            prompts = [{"prompt_token_ids": ids} for ids in chunk.pop("input_ids")]

            outputs = llm.generate(prompts, sampling_params=sampling_params)
            flat_output_ids = [output.token_ids for request_output in outputs for output in request_output.outputs]
            log.info(f"{flat_output_ids[0]}, {cfg.tokenizer.join_with_delimiter}, {cfg.tokenizer.delimiter}")
            flat_decoded = batch_ids_to_string(
                flat_output_ids, tokenizer, cfg.tokenizer.join_with_delimiter, cfg.tokenizer.delimiter
            )

            chunk["decoded_outputs"] = [
                flat_decoded[i : i + cfg.sampling_params.n] for i in range(0, len(flat_decoded), cfg.sampling_params.n)
            ]

            if event_extraction_cfg is not None:
                chunk_results = defaultdict(list)
                for sample_outputs, initial_time, initial_header in zip(
                    chunk["decoded_outputs"], chunk["initial_time"], chunk["initial_header"]
                ):
                    sample_results = defaultdict(list)
                    for out_text in sample_outputs:
                        result = detect_outcome(
                            out_text,
                            code_splitter,
                            delimiter,
                            date_pattern,
                            time_pattern,
                            target_event_regex,
                            end_event_regex,
                            end_time_days,
                            initial_time_offset=initial_time,
                            initial_header=initial_header,
                        )
                        for k, v in result.items():
                            sample_results[k].append(v)

                    sample_results["probability"] = sum(sample_results["outcome_predicted"]) / len(
                        sample_results["outcome_predicted"]
                    )

                    for k, v in sample_results.items():
                        chunk_results[k].append(v)

                chunk.update(chunk_results)

            log.info(f"Saving to {chunk_file_path}")
            table = pa.Table.from_pydict(chunk)
            pq.write_table(table, chunk_file_path)

    except Exception as e:
        task_name = Path(cfg.dataset_dir).name
        batch_info = f" at batch index {i}" if "i" in locals() else ""
        log.error(f"Error processing task {task_name}{batch_info}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
