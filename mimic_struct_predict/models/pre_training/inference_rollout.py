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
    config_path="config/inference_rollout",
    config_name="natural_language",
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model)

    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))
    log.info(sampling_params)

    for task_dir in dataset_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        if selected_tasks is not None and task_name not in selected_tasks:
            log.info(f"Skipping task {task_name} as it is not in the selected tasks list.")
            continue

        log.info(f"Processing Task: [{task_name}]")
        dataset = datasets.load_from_disk(str(task_dir)).shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))

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
                input_ids = chunk["input_ids"]

                max_input_length = cfg.get("max_input_length", None)
                if max_input_length is not None:
                    keep_first = cfg.get("keep_first_token", False)
                    truncated_input_ids = []
                    for ids in input_ids:
                        if len(ids) > max_input_length:
                            if keep_first:
                                truncated_input_ids.append([ids[0]] + ids[-(max_input_length - 1) :])
                            else:
                                truncated_input_ids.append(ids[-max_input_length:])
                        else:
                            truncated_input_ids.append(ids)
                    input_ids = truncated_input_ids

                outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)

                output_ids = [request_output.outputs[0].token_ids for request_output in outputs]

                decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                chunk_dict = dict(chunk)
                chunk_dict["decoded_outputs"] = decoded_outputs

                table = pa.Table.from_pydict(chunk_dict)
                pq.write_table(table, chunk_file_path)

        except Exception as e:
            log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)

    log.info(f"Inference Complete! Outputs saved to {out_dir}")
