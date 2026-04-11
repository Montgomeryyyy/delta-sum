from logging import getLogger

from transformers import PreTrainedTokenizerFast

from mimic_struct_predict.models.utils.prompt_template import create_prompt_str, truncate_on_ehr_newlines

log = getLogger(__name__)


def ehr_texts_to_message_tokens(
    ehr_texts: list[str],
    task_description: str,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
    max_prompt_length: int,
    max_new_tokens: int | None,
) -> dict[str, list]:
    prompts = [
        create_prompt_str(
            # Rough truncation of EHR text with v conservative estimate of 1 token = 15 characters.
            # Final truncation to max_prompt_length will be done after adding task description and thinking.
            ehr_text=ehr_text[-max_prompt_length * 15 :],
            task_description=task_description,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
        )
        for ehr_text in ehr_texts
    ]

    tokenizer_outputs = tokenizer(
        prompts,
        truncation=False,  # Truncation handled by truncate_on_ehr_newlines
        max_length=max_prompt_length,
        padding="do_not_pad",  # We'll handle padding later
        return_length=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    truncated_outputs: dict[str, list] = {"input_ids": [], "attention_mask": [], "length": []}
    for idx, prompt in enumerate(prompts):
        result = truncate_on_ehr_newlines(
            tokenizer_outputs["input_ids"][idx],
            tokenizer_outputs["attention_mask"][idx],
            tokenizer_outputs["offset_mapping"][idx],
            tokenizer_outputs["length"][idx],
            prompt,
            max_prompt_length,
        )
        truncated_outputs["input_ids"].append(result["input_ids"])
        truncated_outputs["attention_mask"].append(result["attention_mask"])
        truncated_outputs["length"].append(result["length"])

    return truncated_outputs


def ehr_texts_to_prompt(
    ehr_texts: list[str],
    task_description: str,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
    max_prompt_length: int,
    max_new_tokens: int | None,
) -> dict[str, list[str]]:
    result = ehr_texts_to_message_tokens(
        ehr_texts=ehr_texts,
        task_description=task_description,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
    )
    return {"prompt": tokenizer.batch_decode(result["input_ids"], skip_special_tokens=False)}
