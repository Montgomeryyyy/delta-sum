import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

NO_PRIOR_CONTEXT = "None."


@dataclass
class PromptJob:
    """One generation request tied back to a ``(patient, chunk)`` location."""

    p_i: int
    c_i: int
    prompt: str
    text: str
    previous_context: str = NO_PRIOR_CONTEXT
    was_truncated: bool = False


def iter_chunks(chunks_col: list[list[dict]]) -> Iterator[tuple[int, int, dict]]:
    for p_i, chunks in enumerate(chunks_col):
        for c_i, chunk in enumerate(chunks):
            yield p_i, c_i, chunk


def format_prompt(
    prompt_template: str,
    tokenizer,
    use_chat_template: bool,
    system_prompt: str | None,
    **fields: str,
) -> str:
    """Build a generation prompt, using the instruct chat template when available."""
    user_content = prompt_template.format(**fields)
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return user_content

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        log.warning(f"apply_chat_template failed ({e}); falling back to raw prompt.")
        return user_content


def format_summarize_prompt(
    text: str,
    prompt_template: str,
    tokenizer,
    use_chat_template: bool,
    system_prompt: str | None,
) -> str:
    return format_prompt(
        prompt_template,
        tokenizer,
        use_chat_template,
        system_prompt,
        text=text,
    )


def content_budget(
    tokenizer,
    template: str,
    system: str | None,
    max_prompt_tokens: int,
    use_chat_template: bool,
    **fields: str,
) -> int:
    """Tokens left for ``fields`` after the instruction wrapper is counted."""
    wrapper_tokens = len(
        tokenizer.encode(
            format_prompt(template, tokenizer, use_chat_template, system, **fields),
            add_special_tokens=False,
        )
    )
    return max(1, max_prompt_tokens - wrapper_tokens)


def generate_summaries(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
    *,
    system_prompt: str | None = None,
) -> list[str]:
    """Run generation over prompts and return decoded texts.

    For local vLLM, ``system_prompt`` is ignored when already baked into prompts via
    chat templates. For the OpenAI backend, it is sent as a chat ``system`` message.
    """
    return llm.generate(prompts, sampling_params, system_prompt=system_prompt)


def truncate_chunk_text(text: str, tokenizer, max_text_tokens: int) -> tuple[str, bool, int]:
    """Keep the last ``max_text_tokens`` of the EHR chunk (not the full prompt).

    Truncating the full prompt would drop the leading instruction for long notes.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)
    if n_tokens <= max_text_tokens:
        return text, False, n_tokens
    return (
        tokenizer.decode(token_ids[-max_text_tokens:], skip_special_tokens=True),
        True,
        n_tokens,
    )


def fit_delta_inputs(
    text: str,
    previous_context: str,
    tokenizer,
    max_content_tokens: int,
) -> tuple[str, str, bool]:
    """Fit the new excerpt and prior context into the content token budget.

    When both do not fit, keep the end of each: the latest part of the new
    excerpt and the most recent prior context.
    """
    prev_ids = tokenizer.encode(previous_context, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(prev_ids) + len(text_ids) <= max_content_tokens:
        return text, previous_context, False

    has_history = previous_context.strip() not in {"", NO_PRIOR_CONTEXT}
    if not has_history:
        prev_budget = min(len(prev_ids), max(0, max_content_tokens - 1))
        text_budget = max_content_tokens - prev_budget
    else:
        # Start from an even split, then give any unused share back to the longer side.
        text_budget = min(len(text_ids), max(1, max_content_tokens // 2))
        prev_budget = min(len(prev_ids), max_content_tokens - text_budget)
        text_budget = min(len(text_ids), max_content_tokens - prev_budget)

    fitted_prev = (
        tokenizer.decode(prev_ids[-prev_budget:], skip_special_tokens=True)
        if prev_budget
        else NO_PRIOR_CONTEXT
    )
    fitted_text = tokenizer.decode(text_ids[-text_budget:], skip_special_tokens=True)
    return fitted_text, fitted_prev, True


def prepare_summary_prompt(
    text: str,
    *,
    tokenizer,
    max_text_tokens: int,
    prompt_template: str,
    use_chat_template: bool,
    system_prompt: str | None,
    subject_id=None,
    chunk_id=None,
) -> tuple[str, str, bool]:
    """Truncate an EHR excerpt and format a standalone summary prompt (no prior context)."""
    text, was_truncated, n_tokens = truncate_chunk_text(text, tokenizer, max_text_tokens)
    if was_truncated:
        log.debug(
            f"Truncated text for subject_id={subject_id} chunk_id={chunk_id}: "
            f"{n_tokens} -> {max_text_tokens} tokens"
        )
    prompt = format_summarize_prompt(
        text, prompt_template, tokenizer, use_chat_template, system_prompt
    )
    return prompt, text, was_truncated


def prepare_context_prompt(
    text: str,
    previous_context: str,
    *,
    tokenizer,
    max_content_tokens: int,
    prompt_template: str,
    use_chat_template: bool,
    system_prompt: str | None,
    subject_id=None,
    chunk_id=None,
) -> tuple[str, str, str, bool]:
    """Fit excerpt + prior context and format a ``{text, previous_context}`` prompt."""
    text, previous_context, was_truncated = fit_delta_inputs(
        text, previous_context, tokenizer, max_content_tokens
    )
    if was_truncated:
        log.debug(
            f"Truncated delta prompt for subject_id={subject_id} chunk_id={chunk_id} "
            f"to {max_content_tokens} content tokens"
        )
    prompt = format_prompt(
        prompt_template,
        tokenizer,
        use_chat_template,
        system_prompt,
        text=text,
        previous_context=previous_context,
    )
    return prompt, text, previous_context, was_truncated

