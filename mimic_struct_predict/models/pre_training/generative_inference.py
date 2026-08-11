"""HuggingFace vLLM inference datasets with dataset sharding for autoregressive rollout."""

import logging
import math
import os
import re
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

DATE_PATTERN = re.compile(r"## Date.*?\+(?P<days>\d+)\s*days")
TIME_PATTERN = re.compile(r"### Time.*?(?P<hours>\d+)h.*?(?P<minutes>\d+)m")
TIME_UNSPECIFIED_HRS = 23
TIME_UNSPECIFIED_MINS = 59


def get_truncate_and_decode_batch_input_ids(
    tokenizer: PreTrainedTokenizerFast,
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

        return {"input_ids": input_ids, "decoded_input": tokenizer.batch_decode(input_ids)}

    return truncate_and_decode_batch_input_ids


def hours_and_mins_to_days(hrs: int, mins: int) -> float:
    return hrs / 24.0 + mins / (24.0 * 60.0)


def get_last_time_offset_and_header(in_text: str, code_splitter: str) -> dict[str, float | str]:
    last_time_offset = None
    last_header = None

    lines = in_text.split(code_splitter)
    for line in reversed(lines):
        line = line.strip()

        if last_header is None and line.startswith("####"):
            last_header = line

        if last_time_offset is None:
            if line.startswith("## Date"):
                last_time_offset = 0.0
            elif line.startswith("### Time"):
                time_match = TIME_PATTERN.search(line)
                if time_match:
                    last_time_offset = hours_and_mins_to_days(
                        int(time_match.group("hours")), int(time_match.group("minutes"))
                    )
                elif line == "### Time Unspecified":
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
    target_event_regex: re.Pattern,
    end_event_regex: re.Pattern | None,
    end_time_days: float | None,
    initial_time_offset: float,
    initial_header: str,
) -> tuple[bool, bool]:
    """Returns (outcome_predicted, format_adherence)."""

    current_time = initial_time_offset
    current_header = initial_header

    try:
        for line in text.split(code_splitter):
            time_updated = False
            line = line.strip()

            if not line:
                continue

            if not line.startswith("#"):
                # It's an event see if it matches with target
                event_str = f"{current_header}{delimiter}{line}"
                if target_event_regex.search(event_str):
                    return True, True
                if end_event_regex is not None and end_event_regex.search(event_str):
                    return False, True
                continue

            if line.startswith("####"):
                current_header = line

            date_match = DATE_PATTERN.search(line)
            if date_match:
                # Move to next N days
                current_time = math.floor(current_time) + int(date_match.group("days"))
                time_updated = True

            time_match = TIME_PATTERN.search(line)
            if time_match:
                hours = int(time_match.group("hours"))
                minutes = int(time_match.group("minutes"))
                current_time = math.floor(current_time) + hours_and_mins_to_days(hours, minutes)
                time_updated = True
            elif line == f"### Time{delimiter}Unspecified":
                current_time = math.floor(current_time) + hours_and_mins_to_days(
                    TIME_UNSPECIFIED_HRS, TIME_UNSPECIFIED_MINS
                )
                time_updated = True

            if time_updated:
                time_covered = current_time - initial_time_offset
                if end_time_days is not None and time_covered > end_time_days:
                    return False, True
                if time_covered < 0:
                    log.warning(f"Negative time covered: {line}")
                    return False, False

        # Reached end of generation without matching regex or exceeding max time.
        return False, False

    except Exception as e:
        log.debug(f"Formatting broken: {e}")
        return False, False


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

    dataset = datasets.load_from_disk(str(dataset_dir)).shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        log.info(f"Running distributed: Rank {rank}/{world_size}. Sharding dataset...")
        dataset = dataset.shard(num_shards=world_size, index=rank)
    log.info(f"Loaded {len(dataset)} examples from task directory (after sharding).")

    ### Preprocessing ###
    # 1. Truncate and decode batch input ids
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

    truncate_and_decode_batch_input_ids = get_truncate_and_decode_batch_input_ids(tokenizer)
    dataset = dataset.map(
        lambda input_ids: truncate_and_decode_batch_input_ids(input_ids, cfg.max_input_length, cfg.keep_first_token),
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
        target_event_regex = re.compile(event_extraction_cfg["target_event_regex"])
        end_event_str = event_extraction_cfg.get("end_event_regex")
        end_event_regex = re.compile(end_event_str) if end_event_str is not None else None
        end_time_days = event_extraction_cfg.get("end_time_days")
        if end_event_regex is None and end_time_days is None:
            raise ValueError("end_event_regex and end_time_days cannot both be None")

        dataset = dataset.map(
            lambda decoded_input: get_last_time_offset_and_header(decoded_input, code_splitter),
            input_columns=["decoded_input"],
            num_proc=cfg.num_workers,
            desc="Extracting last time and header from inputs",
        )

    ### Load model ###
    log.info(f"Loading model ({cfg.model.model})...")
    llm = LLM(**cfg.model)

    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))
    log.info(sampling_params)

    # Batch wise inference and streaming results to Parquet to avoid memory issues with large datasets
    # and allow for partial results to be saved even if the process is interrupted.
    try:
        for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
            chunk_index = i // (cfg.chunk_size)
            log.info(f"Processing Chunk {i + 1}")
            chunk_file_path = inference_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

            # Auto-resume: Skip if this chunk file already exists on disk
            if chunk_file_path.exists():
                continue

            chunk = dict(dataset[i : i + cfg.chunk_size])

            prompts = [{"prompt_token_ids": ids} for ids in chunk.pop("input_ids")]

            outputs = llm.generate(prompts, sampling_params=sampling_params)

            output_ids = [request_output.outputs[0].token_ids for request_output in outputs]
            chunk["decoded_outputs"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            if event_extraction_cfg is not None:
                outcomes = []
                adherences = []

                initial_times = chunk.pop("initial_time")
                initial_headers = chunk.pop("initial_header")

                for out_text, init_time, init_header in zip(chunk["decoded_outputs"], initial_times, initial_headers):
                    outcome_predicted, format_adherence = detect_outcome(
                        out_text,
                        code_splitter,
                        delimiter,
                        target_event_regex,
                        end_event_regex,
                        end_time_days,
                        initial_time_offset=init_time,
                        initial_header=init_header,
                    )
                    outcomes.append(outcome_predicted)
                    adherences.append(format_adherence)

                chunk["outcome_predicted"] = outcomes
                chunk["format_adherence"] = adherences

            log.info(f"Saving to {chunk_file_path}")
            table = pa.Table.from_pydict(chunk)
            pq.write_table(table, chunk_file_path)

    except Exception as e:
        task_name = Path(cfg.dataset_dir).name
        batch_info = f" at batch index {i}" if "i" in locals() else ""
        log.error(f"Error processing task {task_name}{batch_info}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
