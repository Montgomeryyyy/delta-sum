from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForInputTargetLM:
    """Pack separate ``input_ids`` + target ``labels`` for causal LM.

    Dataset rows store input and target separately (for inspection). At train
    time we concatenate them and mask input positions with ``-100`` so loss is
    only on the target tokens. Do not use ``DataCollatorForLanguageModeling`` —
    it would overwrite ``labels`` with a copy of ``input_ids``.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        packed_input_ids: list[list[int]] = []
        packed_labels: list[list[int]] = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            target_ids = [token_id for token_id in feature["labels"] if token_id != -100]
            packed_input_ids.append(input_ids + target_ids)
            packed_labels.append([-100] * len(input_ids) + target_ids)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must have a pad_token_id for input→target training.")

        # Pad right ourselves: tokenizer.pad follows tokenizer.padding_side (left for Gemma),
        # which would misalign the labels written below.
        max_len = max(len(ids) for ids in packed_input_ids)
        if self.pad_to_multiple_of:
            max_len = -(-max_len // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_size = len(packed_input_ids)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        for i, (row_ids, row_labels) in enumerate(zip(packed_input_ids, packed_labels)):
            n = len(row_ids)
            input_ids[i, :n] = torch.tensor(row_ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(row_labels, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForPackedInputTargetLM:
    """Padding-free variant of ``DataCollatorForInputTargetLM``.

    Concatenates every example in the batch into a single row with ``position_ids``
    restarting at 0 per example and no ``attention_mask``. Only valid with
    ``attn_implementation=flash_attention_2``, which turns the position resets into
    varlen boundaries so examples never attend to each other.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        for feature in features:
            context_ids = list(feature["input_ids"])
            target_ids = [token_id for token_id in feature["labels"] if token_id != -100]
            input_ids += context_ids + target_ids
            labels += [-100] * len(context_ids) + target_ids
            position_ids += list(range(len(context_ids) + len(target_ids)))
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "labels": torch.tensor([labels], dtype=torch.long),
            "position_ids": torch.tensor([position_ids], dtype=torch.long),
        }
