import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets
import hydra
from google.genai.local_tokenizer import LocalTokenizer
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from mimic_struct_predict.models.utils import load_tokenizer

log = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Analyze this patient's Electronic Health Record (EHR).

# Patient's EHR
{ehr}

# Task
Based on the patient's electronic health record above, predict if the outcome will occur:
{task_description}

# Instructions
{reasoning_instructions}
{format_instructions}
"""  # noqa: E501

DIRECT_INSTRUCTIONS = """Do not explain your reasoning. Output only your final answer."""

THINKING_INSTRUCTIONS = """Think step by step before giving your final answer.
Your entire response, including thinking, must be less than {max_new_words} words.
"""

YES_NO_FORMAT_INSTRUCTIONS = """Provide your final answer at the very end of your response.
It must be exactly one of the following:
Answer: Yes
OR
Answer: No"""  # noqa: E501

PROBABILITY_FORMAT_INSTRUCTIONS = """Provide your final answer at the very end of your response.
It must be a single decimal value representing the probability of the event occurring (between 0.0 and 1.0).
Example format:
Answer: 0.85
"""  # noqa: E501

OUTPUT_NAME_TO_FORMAT_INSTRUCTIONS = {
    "yes_no": YES_NO_FORMAT_INSTRUCTIONS,
    "probability": PROBABILITY_FORMAT_INSTRUCTIONS,
}


class GeminiTokenizerWrapper:
    """HACK: A simple wrapper around the LocalTokenizer to mimic the HuggingFace PreTrainedTokenizer
    interface purposes."""

    def __init__(self, model: str):
        self.tokenizer = LocalTokenizer(model)

    def __call__(self, contents: list[str], **kwargs: Any) -> dict[str, list[bytes]]:
        # HACK: The LocalTokenizer doesn't have a decode method so use the tokens instead of token ids here.
        return {"input_ids": [token_info.tokens for token_info in self.tokenizer.compute_tokens(contents).tokens_info]}

    def batch_decode(self, batch_token_ids: list[list[bytes]], skip_special_tokens: bool = True) -> list[str]:
        """Rough but gemini tokenizer handles newlines and spaces within tokens, so we can just join the token strings
        together."""
        return [b"".join(token_ids).decode("utf-8") for token_ids in batch_token_ids]

    def apply_chat_template(
        self,
        batch_messages: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[str]:
        if any(len(messages) > 1 for messages in batch_messages):
            raise NotImplementedError(
                "This wrapper only supports simple cases without generation prompts or multi-turn conversations."
            )
        return [messages[0]["content"] for messages in batch_messages]


def truncate_ehr_texts(
    ehr_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_ehr_tokens: int,
    character_per_token_heuristic: int,
) -> list[str]:
    """Truncate a list of EHR texts to fit within the max token limit of the model when
    combined with the prompt instructions."""

    # Character-level pre-slicing to reduce the amount of text we need to tokenize
    # (tokenization is the bottleneck and character slicing is very fast)
    # 6 characters per token is a conservative heuristic
    pre_sliced_ehrs = [ehr[-max_ehr_tokens * character_per_token_heuristic :] for ehr in ehr_texts]

    # Tokenize and left hand truncate
    batch_token_ids = tokenizer(pre_sliced_ehrs, add_special_tokens=False)["input_ids"]
    truncated_batch_ids = [ids[-max_ehr_tokens:] if len(ids) > max_ehr_tokens else ids for ids in batch_token_ids]

    # Decode back to text
    decoded_ehrs = tokenizer.batch_decode(truncated_batch_ids, skip_special_tokens=True)

    # Ideal truncate on a newline (which are very common by design as codes are seperated by \n)
    final_ehrs = []
    for orig_ids, text in zip(batch_token_ids, decoded_ehrs):
        if len(orig_ids) <= max_ehr_tokens:
            final_ehrs.append(text)
        else:
            newline_idx = text.find("\n")
            if newline_idx != -1:
                final_ehrs.append(text[newline_idx + 1 :].lstrip())
            else:
                final_ehrs.append(text)

    return final_ehrs


def create_prompts_from_ehr_texts(
    ehr_texts: list[str],
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    output_format: str,
    max_new_tokens: int | None = None,
) -> list[str]:
    """
    Create the final chat-template prompt string.

    Tokenization and truncation are intentionally handled in separate steps to support
    batch tokenization with Hugging Face tokenizers.

    Inputs:
        ehr_texts (list[str]): A list of patient EHR data as strings.
        task_description (str): A description of the prediction task.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the LLM being used.
        enable_thinking (bool): Whether to include step-by-step reasoning instructions.
        max_new_tokens (int | None): The maximum number of tokens the model is allowed to generate in its response
                                    (required if thinking is enabled, as it's part of the instructions).
    Returns:
        list[str]: A list of prompts rendered via tokenizer chat template.
    Raises:
        ValueError: If `max_new_tokens` is not specified when `enable_thinking` is True, or if
                `max_ehr_tokens` is not greater than 0.

    """
    if output_format not in OUTPUT_NAME_TO_FORMAT_INSTRUCTIONS:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Supported formats: {list(OUTPUT_NAME_TO_FORMAT_INSTRUCTIONS.keys())}"
        )
    format_instructions = OUTPUT_NAME_TO_FORMAT_INSTRUCTIONS[output_format]

    reasoning_instructions = THINKING_INSTRUCTIONS if enable_thinking else DIRECT_INSTRUCTIONS
    if enable_thinking:
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens must be specified when thinking is enabled (as is part of prompt instructions)."
            )
        # Heuristic: 1 word = 0.75 tokens then round down to the nearest 10
        max_new_words = int(((max_new_tokens * 0.75) // 10) * 10)
        reasoning_instructions = reasoning_instructions.format(max_new_words=max_new_words)

    batch_messages = []
    for ehr in ehr_texts:
        formatted_content = PROMPT_TEMPLATE.format(
            ehr=ehr,
            task_description=task_description,
            reasoning_instructions=reasoning_instructions,
            format_instructions=format_instructions,
        )

        messages = [{"role": "user", "content": formatted_content}]
        if not enable_thinking:
            # Add generation prompt to guide the model to output in the correct format for direct prompt

            messages.append({"role": "assistant", "content": "Answer:" if output_format == "yes_no" else "Answer: 0."})

        batch_messages.append(messages)

    prompts: list[str] = tokenizer.apply_chat_template(
        batch_messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=enable_thinking,
        continue_final_message=not enable_thinking,
        tokenize=False,
    )

    return prompts


def ehr_text_to_prompts(
    ehr_texts: list[str],
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    tokenization_cfg: DictConfig,
) -> list[str]:
    """
    Convert a list of EHR texts and boolean values to a list of prompts and completions.

    """
    final_ehrs = truncate_ehr_texts(
        ehr_texts=ehr_texts,
        tokenizer=tokenizer,
        max_ehr_tokens=tokenization_cfg.max_ehr_tokens,
        character_per_token_heuristic=tokenization_cfg.character_per_token_heuristic,
    )

    prompts = create_prompts_from_ehr_texts(
        ehr_texts=final_ehrs,
        task_description=task_description,
        tokenizer=tokenizer,
        enable_thinking=tokenization_cfg.enable_thinking,
        output_format=tokenization_cfg.output_format,
        max_new_tokens=tokenization_cfg.max_new_tokens,
    )

    return prompts


def bool_to_completion(values: list[bool], output_format: str) -> list[str]:
    if output_format == "yes_no":
        return ["Yes" if v else "No" for v in values]
    elif output_format == "probability":
        return ["1" if v else "0" for v in values]
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_prompts_direct",
)
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.is_gemini:
        if not cfg.tokenization.enable_thinking:
            raise ValueError(
                "Gemini models are optimized for reasoning, so thinking instructions "
                "must be enabled when using the GeminiTokenizerWrapper."
            )
        log.info("Using GeminiTokenizerWrapper for local tokenization.")
        tokenizer = GeminiTokenizerWrapper(cfg.model.pretrained_model_name_or_path)
    else:
        tokenizer = load_tokenizer(cfg.model.pretrained_model_name_or_path)

    valid_splits = set(cfg.splits_to_process)
    valid_tasks = set(cfg.get("tasks") or [])

    task_to_split_to_shard_path: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for shard_path in data_dir.glob("*/*/*.parquet"):
        task = shard_path.parent.parent.name
        split = shard_path.parent.name

        if split in valid_splits and (not valid_tasks or task in valid_tasks):
            task_to_split_to_shard_path[task][split].append(str(shard_path))

    log.info(f"Created {len(task_to_split_to_shard_path)} raw tasks datasets.")

    task_to_datasets = {}
    for task, split_to_shards in task_to_split_to_shard_path.items():
        log.info(f"Processing Task: [{task}]")

        try:
            task_description = cfg.task_description[task]
        except KeyError:
            log.error(f"Task description not found for task: {task}")
            raise ValueError(f"Task description not found for task: {task}")

        dataset = datasets.load_dataset("parquet", data_files=split_to_shards)

        # Safely handle and remove empty splits so they don't persist to tokenization/interleaving

        dataset = datasets.DatasetDict({split: ds for split, ds in dataset.items() if len(ds) > 0})

        if len(dataset) == 0:
            log.warning(f"All splits empty for task {task}. Skipping task entirely.")
            continue

        dataset = dataset.map(
            lambda ehr_texts, boolean_values: {
                "prompt": ehr_text_to_prompts(
                    ehr_texts=ehr_texts,
                    task_description=task_description,
                    tokenizer=tokenizer,
                    tokenization_cfg=cfg.tokenization,
                ),
                "completion": bool_to_completion(boolean_values, cfg.tokenization.output_format),
            },
            input_columns=["ehr_text", "boolean_value"],
            remove_columns=["ehr_text"],
            batched=True,
            batch_size=cfg.tokenization.batch_size,
        )

        task_to_datasets[task] = dataset

    log.info(f"Final datasets: {task_to_datasets}")

    for split in valid_splits:
        if split == cfg.train_split_name:
            # Only collect tasks that actually have a valid, populated train split
            train_datasets = [ds[split] for ds in task_to_datasets.values() if split in ds and len(ds[split]) > 0]

            if not train_datasets:
                log.warning("No valid train datasets found across any tasks!")
                continue

            # Interleave train datasets from all tasks to ensure balanced training and shuffle to ensure
            # random mixing of tasks
            train_dataset = datasets.interleave_datasets(
                train_datasets,
                probabilities=[1.0 / len(train_datasets)] * len(train_datasets),
                seed=cfg.random_seed,
                stopping_strategy="all_exhausted",  # Crucial for balancing train gradients
            )
            train_dataset = train_dataset.shuffle(seed=cfg.random_seed)
            train_dataset.save_to_disk(str(output_dir / split))
            log.info(f"Interleaved train dataset: {len(train_dataset)} examples saved to {output_dir / split}")

        else:
            # Keep val/test datasets separate by task to allow for per-task evaluation
            log.info(f"Saving {split} datasets for each task separately...")
            for task, task_ds in task_to_datasets.items():
                if split in task_ds and len(task_ds[split]) > 0:
                    task_ds[split].save_to_disk(str(output_dir / split / task))
                    log.info(f"{task} dataset: {len(task_ds[split])} examples saved to {output_dir / split / task}")
