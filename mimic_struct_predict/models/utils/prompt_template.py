#!/usr/bin/env python
"""Prompt template generation for LLM inference on EHR data."""

import logging
from typing import Any

from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Analyze the following patient Electronic Health Record (EHR) and predict if the following outcome will occur.

# Outcome to Predict
{task_description}

{ehr_title_header}
{ehr}

{instructions_header}
{instructions}
"""  # noqa: E501


EHR_TITLE_HEADER = "# Patient's EHR"
INSTRUCTIONS_HEADER = "# Instructions"

DIRECT_INSTRUCTIONS = """Based on the patient's electronic health record above, predict if the outcome will occur.

## Constraints
You must respond with EXACTLY one word: "Yes" or "No" enclosed in <answer> tags. Do not explain your reasoning.

## Output Format
You must adhere exactly to the following XML output format:
```xml
<answer>
Yes/No
</answer>
```"""  # noqa: E501

THINKING_INSTRUCTIONS = """Based on the patient's electronic health record above, predict the probability that the outcome will occur.

Step 1: Inside of <think> tags, reason step by step if the outcome will occur.
Step 2: Then inside of <answer> tags, respond with "Yes" or "No" if the outcome will occur.

## Constraints
Inside of <answer> tags, you must respond with EXACTLY one word: "Yes" or "No".
Your response assistant message must be less than {max_words} words.

## Output Format
You must adhere exactly to the following XML output format:
```xml
<think>
Detailed clinical reasoning process less than {max_words} words
</think>
<answer>
Yes/No
</answer>
```"""  # noqa: E501


def _char_to_token_idx(offset_mapping: list[tuple[int, int]], char_idx: int, field_name: str) -> int:
    """Map a character index to token index and raise a clear error when not possible."""
    try:
        return next(i for i, (_, end) in enumerate(offset_mapping) if end > char_idx)
    except StopIteration as exc:
        raise ValueError(
            f"Could not map {field_name} character index to token index. "
            "Check tokenizer offset mappings and prompt template format."
        ) from exc


def truncate_on_ehr_newlines(
    input_ids: list[int],
    attention_mask: list[int],
    offset_mapping: list[tuple[int, int]],
    length: int,
    prompt_str: str,
    max_prompt_length: int,
) -> dict[str, Any]:
    """Truncate prompt tokens from the start of the EHR section while preserving the instructions section."""
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be greater than 0.")

    if length != len(input_ids) or length != len(attention_mask):
        raise ValueError("length must match the lengths of input_ids and attention_mask.")

    if length != len(offset_mapping):
        raise ValueError("length must match the length of offset_mapping.")

    if length <= max_prompt_length:
        return {"input_ids": input_ids, "attention_mask": attention_mask, "length": length}

    num_tokens_to_drop = length - max_prompt_length

    # 5. Find newline char of {EHR_TITLE_HEADER} as starting point for truncation
    header = EHR_TITLE_HEADER + "\n"
    header_start_char_idx = prompt_str.find(header)
    if header_start_char_idx == -1:
        raise ValueError("EHR title header not found in prompt. Cannot safely truncate prompt.")

    title_newline_char_idx = header_start_char_idx + len(header) - 1

    instructions_start_char_idx = prompt_str.find("\n" + INSTRUCTIONS_HEADER + "\n", title_newline_char_idx)
    if instructions_start_char_idx == -1:
        raise ValueError("Instructions header not found in prompt. Cannot safely truncate prompt.")

    # Map the character index to the specific token index
    title_newline_token_idx = _char_to_token_idx(offset_mapping, title_newline_char_idx, "EHR title newline")
    instructions_start_token_idx = _char_to_token_idx(
        offset_mapping,
        instructions_start_char_idx,
        "instructions header",
    )

    # Calc earliest truncation point
    truncation_end = title_newline_token_idx + num_tokens_to_drop

    if truncation_end >= instructions_start_token_idx:
        raise ValueError(
            f"Cannot truncate prompt to {max_prompt_length} tokens without cutting into instructions. "
            f"Consider increasing max_prompt_length or reducing the length of instructions/task description."
        )

    # 6. Map back to characters to find the next clean newline
    trunc_end_char_idx = offset_mapping[truncation_end][0]
    next_newline_char_idx = prompt_str.find("\n", trunc_end_char_idx)

    # Map the next newline character index back to its token index
    if next_newline_char_idx != -1 and next_newline_char_idx < instructions_start_char_idx:
        next_newline_token_idx = (
            _char_to_token_idx(
                offset_mapping[truncation_end:],
                next_newline_char_idx,
                "next EHR newline",
            )
            + truncation_end
        )
    else:
        # If no more newlines exist in the EHR section, truncate at exact token cutoff.
        next_newline_token_idx = truncation_end

    # Return the concatenated slices, leaving the middle truncated out
    # +1 ensures we keep the newline of the title, and drop the newline of the cut text
    return {
        "input_ids": input_ids[: title_newline_token_idx + 1] + input_ids[next_newline_token_idx + 1 :],
        "attention_mask": attention_mask[: title_newline_token_idx + 1] + attention_mask[next_newline_token_idx + 1 :],
        "length": length - (next_newline_token_idx - title_newline_token_idx),
    }


def create_prompt_str(
    ehr_text: str,
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    max_prompt_length: int,
    max_new_tokens: int | None = None,
) -> str:
    """
    Create the final chat-template prompt string.

    Tokenization and truncation are intentionally handled in separate steps to support
    batch tokenization with Hugging Face tokenizers.

    Inputs:
        ehr_text (str): The patient's EHR data as a single string.
        task_description (str): A description of the prediction task.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the LLM being used.
        enable_thinking (bool): Whether to include step-by-step reasoning instructions.
        max_prompt_length (int): Validated to be > 0. Enforced during tokenization/truncation.
        max_new_tokens (int | None): The maximum number of tokens the model is allowed to generate in its response
                                    (required if thinking is enabled, as it's part of the instructions).
    Returns:
        str: Prompt rendered via tokenizer chat template.
    Raises:
        ValueError: If `max_new_tokens` is not specified when `enable_thinking` is True, or if
                `max_prompt_length` is not greater than 0.

    """
    if max_prompt_length <= 0:
        raise ValueError("max_prompt_length must be greater than 0.")

    # 1. Recreate the prompt text exactly as the LLM will see it
    instructions = THINKING_INSTRUCTIONS if enable_thinking else DIRECT_INSTRUCTIONS

    if enable_thinking:
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens must be specified when thinking is enabled (as is part of prompt instructions)."
            )
        instructions = instructions.format(max_words=max_new_tokens)

    formatted_content = PROMPT_TEMPLATE.format(
        ehr_title_header=EHR_TITLE_HEADER,
        ehr=ehr_text,
        task_description=task_description,
        instructions_header=INSTRUCTIONS_HEADER,
        instructions=instructions,
    )

    messages = [{"role": "user", "content": formatted_content}]
    if not enable_thinking:
        messages.append(
            {
                "role": "assistant",
                "content": "<answer>\n",
            }
        )

    # 2. Get the prompt as a raw string so we can map characters to tokens
    return str(
        tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=enable_thinking,
            continue_final_message=not enable_thinking,
            tokenize=False,  # Return string instead of token IDs
        )
    )
