"""HuggingFace vLLM inference datasets with dataset sharding for autoregressive rollout."""

import logging
import os
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.sampling_params import StructuredOutputsParams

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """

# Question
{prompt}

# Answer
"""


def get_append_prompt_and_decode_batch_input_ids(
    tokenizer: PreTrainedTokenizerFast,
    prompt_ids: list[int],
) -> Callable[[list[list[int]]], dict[str, Any]]:
    def append_prompt_and_decode_batch_input_ids(batch_input_ids: list[list[int]]) -> dict[str, Any]:
        """Append prompt to batch input ids and decode."""
        input_ids = [ids + prompt_ids for ids in batch_input_ids]
        decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        return {"input_ids": input_ids, "decoded_input": decoded_input}

    return append_prompt_and_decode_batch_input_ids


def extract_yes_no_logit_probabilities(
    request_output: RequestOutput,
    yes_token_id: int,
    no_token_id: int,
    default_prob: float,
) -> tuple[float, bool]:
    """Extract probability of 'Yes' vs 'No' from logits."""
    completion_output = request_output.outputs[0]

    if not completion_output.logprobs:
        raise ValueError("No logprobs found in completion_output. Make sure `logprobs` is > 0 in SamplingParams.")

    last_logprobs = completion_output.logprobs[-1]

    yes_logprob = last_logprobs.get(yes_token_id)
    no_logprob = last_logprobs.get(no_token_id)

    if yes_logprob is None or no_logprob is None:
        return default_prob, False

    # P(Yes) = 1 / (1 + exp(L_no - L_yes))
    prob_yes = 1.0 / (1.0 + np.exp(no_logprob.logprob - yes_logprob.logprob))

    return float(prob_yes), True


@hydra.main(
    version_base=None,
    config_path="config/prompt_inference",
)
def main(cfg: DictConfig) -> None:
    """Run LLM inference and process outputs into Parquet chunks."""

    log.info("Starting LLM Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir)
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(hydra_cfg.run.dir)
    inference_dir = out_dir
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
    log.info(f"Loaded {len(dataset)} examples from data directory (after sharding).")

    ### Preprocessing ###
    # 1. Truncate and decode batch input ids
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

    prompt_ids = tokenizer.encode(
        PROMPT_TEMPLATE.format(prompt=cfg.prompts[cfg.task]),
        add_special_tokens=False,
    )

    append_prompt_and_decode_batch_input_ids = get_append_prompt_and_decode_batch_input_ids(tokenizer, prompt_ids)

    dataset = dataset.map(
        lambda input_ids: append_prompt_and_decode_batch_input_ids(input_ids),
        input_columns=["input_ids"],
        batched=True,
        num_proc=cfg.num_workers,
        desc="Appending prompt to batch input ids and decoding",
    )

    yes_token_id = tokenizer(cfg.yes_token, add_special_tokens=False).input_ids[-1]
    no_token_id = tokenizer(cfg.no_token, add_special_tokens=False).input_ids[-1]
    log.info(f"Using yes token '{cfg.yes_token}' (ID: {yes_token_id}), no token '{cfg.no_token}' (ID: {no_token_id})")

    structured_outputs_params = (
        StructuredOutputsParams(**cfg.structured_outputs) if getattr(cfg, "structured_outputs", None) else None
    )
    sampling_params = SamplingParams(
        structured_outputs=structured_outputs_params, **OmegaConf.to_container(cfg.sampling_params, resolve=True)
    )

    logprobs = getattr(sampling_params, "logprobs", None)
    if logprobs is None or logprobs <= 0:
        raise ValueError(
            "For yes_no output format, `logprobs` must be set to > 0 (large the safer but more memory intenstive) "
            "in `sampling_params` to extract relative yes and no token probabilities."
        )
    max_tokens = getattr(sampling_params, "max_tokens", None)
    if max_tokens is None or max_tokens != 1:
        raise ValueError(
            "For yes_no output format, `max_tokens` must be set to 1 in `sampling_params` to extract relative "
            "yes and no token probabilities."
        )

    ### Load model ###
    log.info(f"Loading model ({cfg.model.model})...")
    cfg.model.max_model_len += len(prompt_ids) + max_tokens
    log.info(f"Adjusted max_model_len to account for prompt and 1 generated token: {cfg.model.max_model_len}")
    llm = LLM(**cfg.model)
    log.info(sampling_params)

    # vLLM inference in chunks
    try:
        for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
            chunk_index = i // (cfg.chunk_size)
            log.info(f"Processing Chunk {chunk_index + 1}  / {len(dataset) // cfg.chunk_size}")
            chunk_file_path = inference_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

            chunk = dict(dataset[i : i + cfg.chunk_size])
            prompts = [{"prompt_token_ids": ids} for ids in chunk.pop("input_ids")]

            outputs = llm.generate(prompts, sampling_params=sampling_params)

            results = [
                extract_yes_no_logit_probabilities(
                    req,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                    default_prob=cfg.default_prob,
                )
                for req in outputs
            ]

            # Unzip results into separate lists
            if results:
                chunk["probability"], chunk["format_adherence"] = map(list, zip(*results))
            else:
                chunk["probability"], chunk["format_adherence"] = [], []
            log.info(f"Saving to {chunk_file_path}")
            table = pa.Table.from_pydict(chunk)
            pq.write_table(table, chunk_file_path)

    except Exception as e:
        batch_info = f" at batch index {i}" if "i" in locals() else ""
        log.error(f"Error processing batch {batch_info}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
