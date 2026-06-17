"""HuggingFace LLM inference datasets with shard-sharding.
Probability extraction supports two modes:
1. yes_no: Model is prompted to end with "Yes" or "No". We extract the logits for these tokens
and compute probability from them.
2. probability: Model is prompted to include a probability in its output, which we extract using a regex.
"""

import logging
import time
from pathlib import Path

import datasets
import hydra
import torch
import transformers
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams

from mimic_struct_predict.models.utils import (
    load_model_and_tokenizer,
)

log = logging.getLogger(__name__)


def batch_inference(
    batch_texts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    generation_config: DictConfig,
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
        truncation=False,
        padding="longest",
        padding_side="left",
        pad_to_multiple_of=64,
        return_tensors="pt",
    ).to(model.device)

    log.info(f"shape {inputs['input_ids'].shape}")

    # Forward pass to get logits for the next token
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            tokenizer=tokenizer,  # Pass tokenizer for handling of stopping criteria
            return_dict_in_generate=True,
            pad_token_id=model.config.pad_token_id,
            **generation_config,
        )

    new_tokens = outputs.sequences[:, inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    end_time = time.perf_counter()
    sample_latency = (end_time - start_time) / len(batch_texts)

    return {
        "completion": completions,
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
    out_dir = Path(hydra_cfg.run.dir)
    log.info(f"Output directory: {out_dir}")

    dataset = datasets.load_from_disk(
        "/scratch/users/k2692250/mimic-struct-predict/data_full/natural_language/prompts/thinking_probability/tuning/ed_to_hospital_admission"
    ).select(range(100))

    if hasattr(transformers.tokenization_utils_base, "PreTrainedTokenizerBase"):
        transformers.tokenization_utils_base.PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )

    llm = LLM(model="Qwen/Qwen3-8B")

    log.info("Loading vLLM model...")
    llm = LLM(
        model=cfg.model.pretrained_model_name_or_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=6000,
        # gpu_memory_utilization=0.9,  # Adjust if you run into Out-Of-Memory issues
    )

    sampling_params = SamplingParams(
        max_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        top_k=cfg.generation.top_k,
        min_p=cfg.generation.min_p,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    prompts = dataset["prompt"]

    _ = llm.generate(prompts[: cfg.batch_size * 3], sampling_params)

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    _ = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    sample_latency = total_time / len(dataset)

    log.info(f"Total execution time: {total_time:.4f} seconds")
    log.info(f"Average sample latency: {sample_latency:.4f} seconds")

    del llm
    torch.cuda.empty_cache()

    # 2. Load model and tokenizer once
    log.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_class=AutoModelForCausalLM,
        model_config=cfg.model,
        eval_mode=True,
        padding_side="left",
    )
    model = torch.compile(model, dynamic=True)

    # Sort to minimize padding within batches
    dataset = dataset.map(
        lambda batch: {"length": [len(prompt) for prompt in batch]},
        input_columns=["prompt"],
        batched=True,
    )
    dataset = dataset.sort("length", reverse=True)
    dataset = dataset.remove_columns("length")

    for i in tqdm(range(0, cfg.batch_size * 3, cfg.batch_size)):
        batch = dataset[i : i + cfg.batch_size]
        batch_inference(
            batch_texts=batch["prompt"],
            model=model,
            tokenizer=tokenizer,
            generation_config=cfg.generation,
        )

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in tqdm(range(0, len(dataset), cfg.batch_size)):
        batch = dataset[i : i + cfg.batch_size]
        batch_inference(
            batch_texts=batch["prompt"],
            model=model,
            tokenizer=tokenizer,
            generation_config=cfg.generation,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    sample_latency = total_time / len(dataset)

    log.info(f"Total execution time: {total_time:.4f} seconds")
    log.info(f"Average sample latency: {sample_latency:.4f} seconds")


if __name__ == "__main__":
    main()
