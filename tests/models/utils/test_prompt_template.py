import pytest

from mimic_struct_predict.models.utils.prompt_template import (
    EHR_TITLE_HEADER,
    create_prompt_str,
    truncate_on_ehr_newlines,
)


class DummyTokenizer:
    """Character-level tokenizer stub for deterministic prompt tests."""

    def __init__(self) -> None:
        self.last_messages: list[dict[str, str]] | None = None
        self.last_apply_kwargs: dict[str, bool] = {}
        self.last_prompt_str: str = ""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
        tokenize: bool,
    ) -> str:
        self.last_messages = messages
        self.last_apply_kwargs = {
            "enable_thinking": enable_thinking,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
            "tokenize": tokenize,
        }

        rendered = "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\n[assistant]\n"

        self.last_prompt_str = rendered
        return rendered

    def __call__(
        self,
        text: str | list[str],
        truncation: bool,
        max_length: int,
        padding: str,
        return_length: bool,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list]:
        self.last_tokenize_kwargs = {
            "truncation": truncation,
            "max_length": max_length,
            "padding": padding,
            "return_length": return_length,
            "add_special_tokens": add_special_tokens,
            "return_offsets_mapping": return_offsets_mapping,
        }

        prompts = [text] if isinstance(text, str) else text
        encoded = [list(range(len(prompt))) for prompt in prompts]

        return {
            "input_ids": encoded,
            "attention_mask": [[1] * len(prompt) for prompt in prompts],
            "offset_mapping": [[(i, i + 1) for i in range(len(prompt))] for prompt in prompts],
            "length": [len(prompt) for prompt in prompts],
        }


def _decode_ids(prompt_str: str, token_ids: list[int]) -> str:
    return "".join(prompt_str[i] for i in token_ids)


def test_create_prompt_str_requires_max_new_tokens_in_thinking_mode() -> None:
    tokenizer = DummyTokenizer()

    with pytest.raises(ValueError, match="max_new_tokens must be specified"):
        create_prompt_str(
            ehr_text="line a",
            task_description="Predict mortality",
            tokenizer=tokenizer,
            enable_thinking=True,
            max_prompt_length=1024,
            max_new_tokens=None,
        )


def test_create_prompt_str_non_thinking_uses_prefilled_answer_tag() -> None:
    tokenizer = DummyTokenizer()

    prompt_str = create_prompt_str(
        ehr_text="line a\nline b",
        task_description="Predict mortality",
        tokenizer=tokenizer,
        enable_thinking=False,
        max_prompt_length=10_000,
    )

    assert tokenizer.last_messages is not None
    assert tokenizer.last_messages[-1] == {"role": "assistant", "content": "<answer>\n"}
    assert tokenizer.last_apply_kwargs == {
        "enable_thinking": False,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "tokenize": False,
    }
    assert prompt_str == tokenizer.last_prompt_str


def test_create_prompt_str_includes_thinking_config_when_enabled() -> None:
    tokenizer = DummyTokenizer()

    prompt_str = create_prompt_str(
        ehr_text="event a",
        task_description="Predict mortality",
        tokenizer=tokenizer,
        enable_thinking=True,
        max_prompt_length=4096,
        max_new_tokens=64,
    )

    assert tokenizer.last_apply_kwargs == {
        "enable_thinking": True,
        "add_generation_prompt": True,
        "continue_final_message": False,
        "tokenize": False,
    }
    assert "less than 64 words" in prompt_str
    assert "[assistant]" in prompt_str


def test_truncate_on_ehr_newlines_truncates_ehr_but_keeps_instructions() -> None:
    tokenizer = DummyTokenizer()
    ehr_text = "\n".join(f"event {i}" for i in range(200))

    prompt_str = create_prompt_str(
        ehr_text=ehr_text,
        task_description="Predict mortality",
        tokenizer=tokenizer,
        enable_thinking=False,
        max_prompt_length=10_000,
    )
    full_prompt = tokenizer(
        prompt_str,
        truncation=False,
        max_length=10_000,
        padding="do_not_pad",
        return_length=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    full_ids = full_prompt["input_ids"][0]
    full_mask = full_prompt["attention_mask"][0]
    full_offsets = full_prompt["offset_mapping"][0]
    full_len = full_prompt["length"][0]

    truncated = truncate_on_ehr_newlines(
        input_ids=full_ids,
        attention_mask=full_mask,
        offset_mapping=full_offsets,
        length=full_len,
        prompt_str=prompt_str,
        max_prompt_length=full_len - 80,
    )

    truncated_text = _decode_ids(prompt_str, truncated["input_ids"])

    assert len(truncated["input_ids"]) <= full_len - 80
    assert EHR_TITLE_HEADER in truncated_text
    assert "# Instructions" in truncated_text
    assert "event 0" not in truncated_text
    assert "event 199" in truncated_text


def test_truncate_on_ehr_newlines_raises_when_header_missing() -> None:
    prompt_str = "no title header\n# Instructions\nfoo"
    full_prompt = list(range(len(prompt_str)))
    offsets = [(i, i + 1) for i in range(len(prompt_str))]

    with pytest.raises(ValueError, match="EHR title header not found"):
        truncate_on_ehr_newlines(
            input_ids=full_prompt,
            attention_mask=full_prompt,
            offset_mapping=offsets,
            length=len(full_prompt),
            prompt_str=prompt_str,
            max_prompt_length=len(full_prompt) - 1,
        )


def test_truncate_on_ehr_newlines_raises_if_it_would_cut_instructions() -> None:
    prompt_str = f"{EHR_TITLE_HEADER}\n" "a\n" "\n" "# Instructions\n" "respond yes or no"
    full_prompt = list(range(len(prompt_str)))
    offsets = [(i, i + 1) for i in range(len(prompt_str))]

    with pytest.raises(ValueError, match="without cutting into instructions"):
        truncate_on_ehr_newlines(
            input_ids=full_prompt,
            attention_mask=full_prompt,
            offset_mapping=offsets,
            length=len(full_prompt),
            prompt_str=prompt_str,
            max_prompt_length=8,
        )


def test_truncate_on_ehr_newlines_raises_on_length_mismatch() -> None:
    prompt_str = f"{EHR_TITLE_HEADER}\na\n\n# Instructions\nrespond"

    with pytest.raises(ValueError, match="length must match the lengths of input_ids and attention_mask"):
        truncate_on_ehr_newlines(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1],
            offset_mapping=[(0, 1), (1, 2), (2, 3)],
            length=3,
            prompt_str=prompt_str,
            max_prompt_length=2,
        )
