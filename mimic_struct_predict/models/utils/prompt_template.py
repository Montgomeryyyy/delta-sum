#!/usr/bin/env python
"""Prompt template generation for LLM inference on EHR data."""

import logging
import re
from enum import StrEnum

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """Analyze the following patient Electronic Health Record (EHR) and predict if the following outcome will occur.

# Outcome to Predict
{task_description}

# Patient's EHR
{ehr}

# Instructions
{instructions}
"""  # noqa: E501


class PromptMode(StrEnum):
    """Supported prompt modes for LLM inference."""

    DIRECT = "direct"
    THINKING = "thinking"


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

# Regex to match age headers: ## Age X years Y days
AGE_HEADER_PATTERN = re.compile(r"^## Age \d+ years \d+ days$", re.MULTILINE)


def ehr_text_to_prompt(
    ehr_text: str,
    task_description: str,
    instructions: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
) -> str:
    """Convert EHR text to a formatted prompt.

    Args:
        ehr_text: The patient's electronic health record in markdown format.
        task_description: Description of the prediction task.
        instructions: Instructions string.
        tokenizer: Tokenizer for chat templating.
        enable_thinking: Whether to enable thinking in the prompt.

    Returns:
        Formatted prompt string ready for LLM input.
    """

    formatted_content = PROMPT_TEMPLATE.format(
        ehr=ehr_text,
        task_description=task_description,
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

    prompt = tokenizer.apply_chat_template(
        messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=enable_thinking,
        continue_final_message=not enable_thinking,
        tokenize=False,
    )

    return str(prompt)


def truncate_ehr(
    ehr_text: str,
    instructions: str,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    task_description: str,
    enable_thinking: bool,
) -> str:
    """Truncate EHR text at age header boundaries to fit within token limit.

    Args:
        ehr_text: The patient's EHR in markdown format.
        instructions: Instructions string.
        tokenizer: HuggingFace tokenizer for counting tokens.
        max_prompt_length: Maximum total sequence length (context window).
        task_description: Task description.
        enable_thinking: Whether to enable thinking in the prompt.

    Returns:
        Truncated EHR text that fits within the token limit.
    """
    # Calculate lengths based on the actual full prompt
    # 1. Check if the full text already fits
    full_prompt = ehr_text_to_prompt(
        ehr_text,
        task_description,
        instructions,
        tokenizer,
        enable_thinking,
    )
    full_tokens = tokenizer.tokenize(full_prompt)

    if len(full_tokens) <= max_prompt_length:
        return ehr_text

    # 2. Derive available tokens correctly
    ehr_tokens = tokenizer.tokenize(ehr_text)
    num_tokens_to_drop = len(full_tokens) - max_prompt_length
    available_tokens = len(ehr_tokens) - num_tokens_to_drop

    if available_tokens <= 0:
        # This means even dropping ALL EHR tokens isn't enough (overhead > max_len)
        overhead_estimate = len(full_tokens) - len(ehr_tokens)
        raise ValueError(
            f"Available tokens ({available_tokens}) <= 0. "
            f"max_prompt_length={max_prompt_length}, "
            f"estimated_overhead={overhead_estimate}, "
            f"ehr_tokens={len(ehr_tokens)}"
        )

    # 3. Try to truncate to age header using regex
    ehr_suffix = str(tokenizer.convert_tokens_to_string(ehr_tokens[-available_tokens:]))

    match = AGE_HEADER_PATTERN.search(ehr_suffix)
    if match:
        return ehr_suffix[match.start() :]

    # Last resort: truncate tokens from the end
    logger.warning(
        f"Could not find a compatible header to truncate on. "
        f"EHR tokens: {len(ehr_tokens)}, Available tokens: {available_tokens}. "
        f"Max prompt length: {max_prompt_length}. Truncating from end."
    )
    return ehr_suffix


def create_prompt(
    ehr_text: str,
    task_description: str,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    max_prompt_length: int | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """Create a formatted prompt from EHR text and task description.

    Args:
        ehr_text: The patient's electronic health record in markdown format.
        task_description: Description of the prediction task.
        tokenizer: Tokenizer for truncation and chat templating.
        mode: Prompt mode selecting which instruction template to use.
        max_prompt_length: Maximum total sequence length (context window).
        max_new_tokens: Maximum new tokens (required for THINKING mode).

    Returns:
        Formatted prompt string ready for LLM input.
    """
    instructions = THINKING_INSTRUCTIONS if enable_thinking else DIRECT_INSTRUCTIONS

    if enable_thinking:
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens must be specified when mode is "
                "PromptMode.THINKING_YES_NO or PromptMode.THINKING_PROB"
            )
        instructions = instructions.format(max_words=max_new_tokens)

    if max_prompt_length is not None:
        ehr_text = truncate_ehr(
            ehr_text,
            instructions,
            tokenizer,
            max_prompt_length,
            task_description,
            enable_thinking,
        )

    return ehr_text_to_prompt(
        ehr_text,
        task_description,
        instructions,
        tokenizer,
        enable_thinking,
    )
