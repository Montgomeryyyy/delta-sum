# import asyncio
# import logging
# import re
# from collections import defaultdict
# from pathlib import Path

# import datasets
# import hydra
# import pyarrow as pa
# import pyarrow.parquet as pq
# from aiolimiter import AsyncLimiter
# from google import genai
# from google.genai.local_tokenizer import LocalTokenizer
# from google.genai.types import AutomaticFunctionCallingConfig, GenerateContentConfig, ThinkingConfig
# from hydra.core.hydra_config import HydraConfig
# from omegaconf import DictConfig
# from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential
# from tqdm.asyncio import tqdm

# from mimic_struct_predict.models.natural_language.inference import extract_probability_by_regex

# log = logging.getLogger(__name__)


# @retry(
#     retry=retry_if_exception_type(genai.errors.APIError),
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=4, max=60),
#     before_sleep=before_sleep_log(log, logging.WARNING),
# )
# async def gemini_inference(
#     prompt: str,
#     boolean_value: bool,
#     model_name: str,
#     client: genai.Client,
#     generation_config: GenerateContentConfig,
#     tokenizer: LocalTokenizer,
#     regex_pattern: re.Pattern,
#     default_value: float,
#     rpm_limit: AsyncLimiter,
#     tpm_limit: AsyncLimiter,
# ) -> dict:
#     """Run a single Gemini request governed strictly by RPM/TPM limits."""
#     # Assuming no system messages and a single user message, so total tokens is just the prompt tokens.
#     # Adjust if this assumption doesn't hold.
#     input_tokens = tokenizer.count_tokens(prompt).total_tokens
#     async with rpm_limit:
#         await tpm_limit.acquire(input_tokens)
#         response = await client.aio.models.generate_content(
#             model=model_name,
#             contents=prompt,
#             config=generation_config,
#         )

#     if response.candidates is None or len(response.candidates) == 0:
#         log.warning("No candidates returned")
#         completion = ""
#         thoughts = ""
#         adherence = 0.0
#         prob = default_value
#     else:
#         parts = response.candidates[0].content.parts
#         # This is not full thinking trace but a summary of it
#         thoughts = "\n".join([part.text for part in parts if part.thought and part.text])
#         completion = "\n".join([part.text for part in parts if not part.thought and part.text])
#         probs, adherences = extract_probability_by_regex([completion], None, regex_pattern, default_value)
#         prob = probs[0]
#         adherence = adherences[0]

#     # Return the original example updated with model results
#     return {
#         "prompt": prompt,
#         "boolean_value": boolean_value,
#         "completion": completion,
#         "thoughts": thoughts,
#         "probability": prob,
#         "format_adherence": adherence,
#         "prompt_token_count": response.usage_metadata.prompt_token_count,
#         "thoughts_token_count": response.usage_metadata.thoughts_token_count,
#         "candidates_token_count": response.usage_metadata.candidates_token_count,
#     }


# async def async_main(cfg: DictConfig) -> None:
#     log.info("Starting LLM Inference")
#     hydra_cfg = HydraConfig.get()
#     out_dir = Path(hydra_cfg.run.dir)

#     # Setup Configs
#     model_name = cfg.model.name
#     tokenizer = LocalTokenizer(model_name)
#     regex_pattern = re.compile(cfg.post_processing.regex_pattern)

#     generation_config = GenerateContentConfig(
#         thinking_config=ThinkingConfig(**cfg.thinking),
#         # We aren't using function calling and creates annoying log message if not disabled
#         automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
#         **cfg.generation,
#     )
#     default_prob = cfg.post_processing.default_prob

#     # Rate Limiters (The "Only" Limiters)
#     rpm_limit = AsyncLimiter(cfg.quotas.rpm_limit, 60)
#     tpm_limit = AsyncLimiter(cfg.quotas.tpm_limit, 60)

#     dataset_dir = Path(cfg.dataset_dir) / cfg.split
#     tasks = cfg.get("tasks", None)
#     split = cfg.split
#     client = genai.Client(vertexai=True, location=cfg.client.location)

#     try:
#         for task_dir in dataset_dir.iterdir():
#             if not task_dir.is_dir() or (tasks and task_dir.name not in tasks):
#                 continue

#             log.info(f"Processing Task: [{task_dir.name}]")
#             dataset = datasets.load_from_disk(str(task_dir))
#             log.info(f"Loaded {len(dataset)} examples from task directory.")

#             writer = None
#             out_file_path = out_dir / split / task_dir.name / "results.parquet"
#             out_file_path.parent.mkdir(parents=True, exist_ok=True)
#             log.info(f"Running inference and streaming to {out_file_path}")
#             try:
#                 for i in tqdm(range(0, len(dataset), cfg.batch_size)):
#                     batch = dataset[i : i + cfg.batch_size]
#                     batch_tasks = (
#                         gemini_inference(
#                             prompt=prompt,
#                             boolean_value=boolean_value,
#                             model_name=model_name,
#                             tokenizer=tokenizer,
#                             client=client,
#                             regex_pattern=regex_pattern,
#                             default_value=default_prob,
#                             generation_config=generation_config,
#                             rpm_limit=rpm_limit,
#                             tpm_limit=tpm_limit,
#                         )
#                         for prompt, boolean_value in zip(batch["prompt"], batch["boolean_value"])
#                     )

#                     batch_results = defaultdict(list)
#                     for future in asyncio.as_completed(batch_tasks):
#                         result_row = await future
#                         for key, values in result_row.items():
#                             batch_results[key].append(values)

#                     table = pa.Table.from_pydict(batch_results)
#                     if writer is None:
#                         writer = pq.ParquetWriter(out_file_path, table.schema)
#                     writer.write_table(table)

#             except Exception as e:
#                 log.error(f"Error processing task {task_dir.name} at batch index {i}: {e}", exc_info=True)
#             finally:
#                 if writer is not None:
#                     writer.close()
#     finally:
#         await client.aio.aclose()


# @hydra.main(version_base=None, config_path="config", config_name="inference_gemini")
# def main(cfg: DictConfig) -> None:
#     asyncio.run(async_main(cfg))
