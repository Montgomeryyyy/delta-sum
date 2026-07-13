"""HuggingFace vLLM inference datasets with dataset sharding.
Probability extraction modes:
1. yes_no: Model outputs "Yes" or "No". Extracts logits to compute probability.
2. probability: Model outputs a probability, extracted via regex.
"""

import logging
import os
import re
from pathlib import Path

import datasets
import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput

log = logging.getLogger(__name__)


def extract_yes_no_self_consistency_probabilities(
    request_output: RequestOutput,
    regex_pattern: re.Pattern,
    default_prob: float = 0.5,
) -> tuple[float, bool]:
    """Extract probability of 'Yes' vs 'No' using self-consistency over multiple sampled completions."""

    yes_count, total_count = 0, 0
    for completion_output in request_output.outputs:
        match = re.search(regex_pattern, completion_output.text)
        if match is None:
            continue

        try:
            extracted_value = match.group(1).strip()
            if extracted_value == "Yes":
                yes_count += 1
        except IndexError:
            if match.group(0).strip().endswith("Yes"):
                yes_count += 1

        total_count += 1

    if total_count == 0:
        return default_prob, False

    return yes_count / total_count, True


def extract_yes_no_logit_probabilities(
    request_output: RequestOutput,
    yes_token_id: int,
    no_token_id: int,
    regex_pattern: re.Pattern,
    default_prob: float = 0.5,
) -> tuple[float, bool]:
    """Extract probability of 'Yes' vs 'No' from logits, assuming the output ends with 'Yes' or 'No'."""
    completion_output = request_output.outputs[0]
    if re.search(regex_pattern, completion_output.text) is None:
        return default_prob, False

    if completion_output.logprobs is None or len(completion_output.logprobs) == 0:
        raise ValueError("No logprobs found in completion_output. Make sure `logprobs` is set to -1 in SamplingParams.")

    last_logprobs = completion_output.logprobs[-1]
    yes_logprob_obj = last_logprobs.get(yes_token_id)
    no_logprob_obj = last_logprobs.get(no_token_id)

    # In empircal testing always returned as option 1 and 2
    if yes_logprob_obj is None:
        log.info("Yes log_prob not returned in top k so setting to 0")
        yes_logprob_obj = Logprob(logprob=0.0)
    if no_logprob_obj is None:
        log.info("No log_prob not returned in top k so setting to 0")
        no_logprob_obj = Logprob(logprob=0.0)

    # P(Yes) = p_yes / (p_yes + p_no) = 1 / (1 + (p_no / p_yes)) = 1 / (1 + exp(L_no - L_yes))
    prob_yes = 1 / (1 + np.exp(no_logprob_obj.logprob - yes_logprob_obj.logprob))

    return float(prob_yes), True


def extract_probability_by_regex(
    request_output: RequestOutput,
    regex_pattern: re.Pattern,
    default_prob: float,
) -> tuple[float, bool]:
    """Extract probability from text using a regular expression."""
    # Extract sequences and logits
    completion_output = request_output.outputs[0]
    match = re.search(regex_pattern, completion_output.text)
    probability, format_adherence = default_prob, False
    if match:
        try:
            # Strip whitespace and convert to float
            prob_str = match.group(1).strip()
            probability = float(prob_str)
            format_adherence = True
        except (ValueError, IndexError):
            pass  # Keep default probability and format adherence False if conversion fails

    return probability, format_adherence


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="inference_direct",
)
def main(cfg: DictConfig) -> None:
    """Run LLM inference and process outputs into Parquet chunks."""

    log.info("Starting LLM Inference")
    hydra_cfg = HydraConfig.get()
    dataset_dir = Path(cfg.dataset_dir) / cfg.split
    selected_tasks = cfg.get("tasks", None)
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")

    # 2. Load model and tokenizer once
    log.info("Loading model...")
    llm = LLM(**cfg.model)
    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))

    compiled_regex = re.compile(cfg.post_processing.regex_pattern)

    if cfg.post_processing.output_format == "yes_no":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)
        yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[-1]
        no_token_id = tokenizer("No", add_special_tokens=False).input_ids[-1]
        logprobs = getattr(sampling_params, "logprobs", None)
        if logprobs is None or logprobs <= 0:
            raise ValueError(
                "For yes_no output format, `logprobs` must be set to > 0 (large the safer but more memory intenstive) "
                "in `sampling_params` to extract relative yes and no token probabilities."
            )

        def process_output(comp_output: RequestOutput) -> tuple[float, bool]:
            return extract_yes_no_logit_probabilities(
                comp_output,
                yes_token_id,
                no_token_id,
                compiled_regex,
                cfg.post_processing.default_prob,
            )
    elif cfg.post_processing.output_format == "yes_no_self_consistency":

        def process_output(comp_output: RequestOutput) -> tuple[float, bool]:
            return extract_yes_no_self_consistency_probabilities(
                comp_output,
                compiled_regex,
                cfg.post_processing.default_prob,
            )
    elif cfg.post_processing.output_format == "probability":

        def process_output(comp_output: RequestOutput) -> tuple[float, bool]:
            return extract_probability_by_regex(
                comp_output,
                compiled_regex,
                cfg.post_processing.default_prob,
            )
    else:
        raise NotImplementedError(
            f"Post_processing format {cfg.post_processing.output_format} is not supported "
            "please choose 'probability' or 'yes_no'"
        )

    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        dataset = datasets.load_from_disk(str(task_dir))

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        if world_size > 1:
            log.info(f"Running distributed: Rank {rank}/{world_size}. Sharding dataset...")
            dataset = dataset.shard(num_shards=world_size, index=rank)

        log.info(f"Loaded {len(dataset)} examples from task directory (after sharding).")

        # Batch wise inference and streaming results to Parquet to avoid memory issues with large datasets
        # and allow for partial results to be saved even if the process is interrupted.
        inference_dir = out_dir / cfg.split / task_dir.name
        inference_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Running inference and streaming to {inference_dir}")

        try:
            for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
                chunk_index = i // (cfg.chunk_size)
                log.info(f"Processing Chunk {i + 1}")
                chunk_file_path = inference_dir / f"chunk_rank{rank:02d}_{chunk_index:06d}.parquet"

                # Auto-resume: Skip if this chunk file already exists on disk
                if chunk_file_path.exists():
                    continue

                chunk = dataset[i : i + cfg.chunk_size]
                prompts = chunk["prompt"]
                outputs = llm.generate(prompts, sampling_params)

                probs, format_adherences, completions = [], [], []
                for request_output in tqdm(outputs, desc="Post processing"):
                    prob, format_adherence = process_output(request_output)
                    probs.append(prob)
                    format_adherences.append(format_adherence)
                    completions.append(request_output.outputs[0].text)

                table = pa.Table.from_pydict(
                    {
                        "prompt": prompts,
                        "boolean_value": chunk["boolean_value"],
                        "completion": completions,
                        "prob": probs,
                        "format_adherence": format_adherences,
                    }
                )
                pq.write_table(table, chunk_file_path)

        except Exception as e:
            log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
