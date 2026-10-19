import logging

from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)

NO_PRIOR_DELTAS = "None."


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


def generate_summaries(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    """Run vLLM generation over prompts and return decoded texts."""
    if not prompts:
        return []
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


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
    previous_deltas: str,
    tokenizer,
    max_content_tokens: int,
) -> tuple[str, str, bool]:
    """Fit the new excerpt and prior updates into the content token budget.

    When both do not fit, keep the end of each: the latest part of the new
    excerpt and the most recent prior updates.
    """
    prev_ids = tokenizer.encode(previous_deltas, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(prev_ids) + len(text_ids) <= max_content_tokens:
        return text, previous_deltas, False

    has_history = previous_deltas.strip() not in {"", NO_PRIOR_DELTAS}
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
        else NO_PRIOR_DELTAS
    )
    fitted_text = tokenizer.decode(text_ids[-text_budget:], skip_special_tokens=True)
    return fitted_text, fitted_prev, True
