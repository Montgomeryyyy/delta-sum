"""HuggingFace vLLM inference datasets with dataset sharding for autoregressive rollout."""

import logging
import os
from pathlib import Path

import datasets
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


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

    # 2. Load model and tokenizer once
    log.info(f"Loading model ({cfg.model.model})...")
    llm = LLM(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

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

            chunk = dataset[i : i + cfg.chunk_size]
            input_ids = chunk["input_ids"]

            max_input_length = cfg.get("max_input_length", None)
            keep_first = cfg.get("keep_first_token", False)

            prompts = []
            for ids in input_ids:
                if max_input_length is not None and len(ids) > max_input_length:
                    if keep_first:
                        ids = [ids[0]] + ids[-(max_input_length - 1) :]
                    else:
                        ids = ids[-max_input_length:]
                prompts.append({"prompt_token_ids": ids})

            outputs = llm.generate(prompts, sampling_params=sampling_params)

            output_ids = [request_output.outputs[0].token_ids for request_output in outputs]

            decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            chunk_dict = dict(chunk)
            chunk_dict.pop("input_ids")
            chunk_dict["decode_inputs"] = decoded_inputs
            chunk_dict["decoded_outputs"] = decoded_outputs
            log.info(f"Saving to {chunk_file_path}")
            table = pa.Table.from_pydict(chunk_dict)
            pq.write_table(table, chunk_file_path)

    except Exception as e:
        task_name = Path(cfg.dataset_dir).name
        batch_info = f" at batch index {i}" if "i" in locals() else ""
        log.error(f"Error processing task {task_name}{batch_info}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
