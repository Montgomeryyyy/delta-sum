"""HuggingFace vLLM inference datasets with dataset sharding.
Probability extraction modes:
1. yes_no: Model outputs "Yes" or "No". Extracts logits to compute probability.
2. probability: Model outputs a probability, extracted via regex.
"""

import logging
import re
from pathlib import Path

import datasets
import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput

log = logging.getLogger(__name__)


def extract_yes_no_logit_probabilities(
    completion_output: CompletionOutput,
    yes_token_id: int,
    no_token_id: int,
    regex_pattern: re.Pattern,
    default_prob: float = 0.5,
) -> tuple[float, bool]:
    """Extract probability of 'Yes' vs 'No' from logits, assuming the output ends with 'Yes' or 'No'."""
    if re.search(regex_pattern, completion_output.text) is None:
        return default_prob, False

    if completion_output.logprobs is None or len(completion_output.logprobs) == 0:
        raise ValueError("No logprobs found in completion_output. Make sure `logprobs` is set to -1 in SamplingParams.")

    last_logprobs = completion_output.logprobs[-1]
    yes_logprob_obj = last_logprobs.get(yes_token_id)
    no_logprob_obj = last_logprobs.get(no_token_id)

    if yes_logprob_obj is None or no_logprob_obj is None:
        raise ValueError("Yes or No token ID was not found in logprobs.")

    # P(Yes) = p_yes / (p_yes + p_no) = 1 / (1 + (p_no / p_yes)) = 1 / (1 + exp(L_no - L_yes))
    prob_yes = 1 / (1 + np.exp(yes_logprob_obj.logprob - no_logprob_obj.logprob))

    return float(prob_yes), True


def extract_probability_by_regex(
    output: CompletionOutput,
    regex_pattern: re.Pattern,
    default_prob: float,
) -> tuple[float, bool]:
    """Extract probability from text using a regular expression."""
    # Extract sequences and logits

    match = re.search(regex_pattern, output.text)
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

    sampling_params = SamplingParams(**cfg.sampling_params)

    if cfg.post_processing.output_format == "yes_no":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)
        yes_token_id = tokenizer("Yes", add_special_tokens=False).input_ids[-1]
        no_token_id = tokenizer("No", add_special_tokens=False).input_ids[-1]
        if sampling_params.logprobs != -1:
            raise ValueError(
                "For yes_no output format, `logprobs` must be set to -1 (whole vocab) "
                "in `sampling_params` to guarantee yes and no token log probabilities are extracted."
            )

    # Pre-compile regex and configure the processing function to avoid inner-loop overhead
    compiled_regex = re.compile(cfg.post_processing.regex_pattern)

    if cfg.post_processing.output_format == "yes_no":

        def process_output(comp_output: CompletionOutput) -> tuple[float, bool]:
            return extract_yes_no_logit_probabilities(
                comp_output,
                yes_token_id,
                no_token_id,
                compiled_regex,
                cfg.post_processing.default_prob,
            )
    elif cfg.post_processing.output_format == "probability":

        def process_output(comp_output: CompletionOutput) -> tuple[float, bool]:
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
        # TODO: Remove when running on full dataset
        dataset = datasets.load_from_disk(str(task_dir)).select(range(1000))
        log.info(f"Loaded {len(dataset)} examples from task directory.")

        # Batch wise inference and streaming results to Parquet to avoid memory issues with large datasets
        # and allow for partial results to be saved even if the process is interrupted.
        inference_dir = out_dir / cfg.split / task_dir.name
        inference_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Running inference and streaming to {inference_dir}")

        try:
            for i in tqdm(range(0, len(dataset), cfg.chunk_size)):
                chunk_index = i // (cfg.chunk_size)
                chunk_file_path = inference_dir / f"chunk_{chunk_index:06d}.parquet"

                # Auto-resume: Skip if this chunk file already exists on disk
                if chunk_file_path.exists():
                    continue

                chunk = dataset[i : i + cfg.chunk_size]
                prompts = chunk["prompt"]
                outputs = llm.generate(prompts, sampling_params)

                probs, format_adherences = [], []
                for request_output in outputs:
                    completion_output = request_output.outputs[0]
                    prob, format_adherence = process_output(completion_output)
                    probs.append(prob)
                    format_adherences.append(format_adherence)

                table = pa.Table.from_pydict(
                    {
                        "prompt": prompts,
                        "boolean_value": chunk["boolean_value"],
                        "completion": [request_output.outputs[0].text for request_output in outputs],
                        "prob": probs,
                        "format_adherence": format_adherences,
                    }
                )
                pq.write_table(table, chunk_file_path)

        except Exception as e:
            log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)

    log.info("Inference Complete!")
