import logging
from collections import defaultdict
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from mimic_struct_predict.models.utils import load_tokenizer

log = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Analyze the following patient Electronic Health Record (EHR) and predict if the following outcome will occur.

# Outcome to Predict
{task_description}

# Patient's EHR"
{ehr}

"# Instructions"
{instructions}
"""  # noqa: E501

DIRECT_INSTRUCTIONS = """Based on the patient's electronic health record above, predict if the outcome will occur.

## Constraints
Do not explain your reasoning.
Your response must start with "Answer: " followed by either "Yes" or "No".

## Output Format
Answer: Yes/No
"""  # noqa: E501

THINKING_INSTRUCTIONS = """Based on the patient's electronic health record above, predict if the outcome will occur.

First, write out your step-by-step clinical reasoning inside <think> and </think> tags.
After you have finished thinking, you must provide your final answer.

## Constraints
Your final answer must start with "Answer: " followed by either "Yes" or "No".
Your entire response, including thinking, must be less than {max_tokens} tokens.

## Output Format
Answer: Yes/No
"""  # noqa: E501


def truncate_ehr_texts(
    ehr_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    empty_prompt_length: int,
    buffer: int,
) -> list[str]:
    """Truncate a list of EHR texts to fit within the max token limit of the model when
    combined with the prompt instructions."""

    max_ehr_tokens = max_prompt_length - empty_prompt_length - buffer
    if max_ehr_tokens <= 0:
        raise ValueError(f"Max prompt length ({max_prompt_length}) is too small to fit instructions.")

    # Character-level pre-slicing to reduce the amount of text we need to tokenize
    # (tokenization is the bottleneck and character slicing is very fast)
    # 6 characters per token is a conservative heuristic
    pre_sliced_ehrs = [ehr[-max_ehr_tokens * 6 :] for ehr in ehr_texts]

    # Tokenize and left hand truncate
    batch_token_ids = tokenizer(pre_sliced_ehrs, add_special_tokens=False)["input_ids"]
    truncated_batch_ids = [ids[-max_ehr_tokens:] if len(ids) > max_ehr_tokens else ids for ids in batch_token_ids]

    # Decode back to text
    decoded_ehrs = tokenizer.batch_decode(truncated_batch_ids, skip_special_tokens=True)

    # Ideal truncate on a newline
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
                `max_prompt_length` is not greater than 0.

    """
    # 1. Recreate the prompt text exactly as the LLM will see it
    instructions = THINKING_INSTRUCTIONS if enable_thinking else DIRECT_INSTRUCTIONS

    if enable_thinking:
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens must be specified when thinking is enabled (as is part of prompt instructions)."
            )
        instructions = instructions.format(max_tokens=max_new_tokens)

    batch_messages = []
    for ehr in ehr_texts:
        formatted_content = PROMPT_TEMPLATE.format(
            ehr=ehr,
            task_description=task_description,
            instructions=instructions,
        )

        messages = [{"role": "user", "content": formatted_content}]
        if not enable_thinking:
            messages.append({"role": "assistant", "content": "Answer:"})

        batch_messages.append(messages)

    prompts: list[str] = tokenizer.apply_chat_template(
        batch_messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=enable_thinking,
        continue_final_message=not enable_thinking,
        tokenize=False,
    )

    return prompts


def ehr_text_and_boolean_to_prompts_and_completions(
    ehr_texts: list[str],
    boolean_values: list[bool],
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    max_prompt_length: int,
    empty_prompt_length: int,
    yes_token: str,
    no_token: str,
    max_new_tokens: int | None = None,
    buffer: int = 16,
) -> dict[str, list[str]]:
    """
    Convert a list of EHR texts and boolean values to a list of prompts and completions.

    """

    final_ehrs = truncate_ehr_texts(
        ehr_texts=ehr_texts,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        empty_prompt_length=empty_prompt_length,
        buffer=buffer,
    )

    prompts = create_prompts_from_ehr_texts(
        ehr_texts=final_ehrs,
        task_description=task_description,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
    )

    completions = [yes_token if boolean_value else no_token for boolean_value in boolean_values]

    return {"prompt": prompts, "completion": completions}


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="create_prompts_direct",
)
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        dataset = datasets.load_dataset("parquet", data_files=split_to_shards, columns=["ehr_text", "boolean_value"])

        # Safely handle and remove empty splits so they don't persist to tokenization/interleaving

        dataset = datasets.DatasetDict({split: ds for split, ds in dataset.items() if len(ds) > 0})

        if len(dataset) == 0:
            log.warning(f"All splits empty for task {task}. Skipping task entirely.")
            continue

        empty_prompt = create_prompts_from_ehr_texts(
            ehr_texts=[""],
            task_description=task_description,
            tokenizer=tokenizer,
            enable_thinking=cfg.tokenization.enable_thinking,
            max_new_tokens=cfg.tokenization.max_new_tokens,
        )[0]
        empty_prompt_length = len(tokenizer.tokenize(empty_prompt, add_special_tokens=False))

        dataset = dataset.map(
            lambda ehr_texts, boolean_values: ehr_text_and_boolean_to_prompts_and_completions(
                ehr_texts=ehr_texts,
                boolean_values=boolean_values,
                task_description=task_description,
                tokenizer=tokenizer,
                max_prompt_length=cfg.tokenization.max_prompt_length,
                enable_thinking=cfg.tokenization.enable_thinking,
                max_new_tokens=cfg.tokenization.max_new_tokens,
                empty_prompt_length=empty_prompt_length,
                yes_token=cfg.tokenization.yes_token,
                no_token=cfg.tokenization.no_token,
                buffer=cfg.tokenization.token_buffer,
            ),
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
