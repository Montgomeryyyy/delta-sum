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

        batch = self.tokenizer.pad(
            {
                "input_ids": packed_input_ids,
                "attention_mask": [[1] * len(ids) for ids in packed_input_ids],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id for input→target training.")

        labels = torch.full((len(packed_labels), max_len), fill_value=-100, dtype=torch.long)
        for i, row_labels in enumerate(packed_labels):
            labels[i, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
        batch["labels"] = labels
        return batch
