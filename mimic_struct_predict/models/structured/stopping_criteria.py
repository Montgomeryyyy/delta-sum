from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


class WordLevelStopCriteria(StoppingCriteria):
    """Custom stopping criteria that checks for multiple stop strings across a batch of sequences.
    Designed to work with WordLevel tokenizers where stop strings are tokenized into fixed ID sequences.
    The generation will stop once every sequence in the batch has generated at least one of the specified stop strings.
    """

    def __init__(self, stop_strings: list[str], tokenizer: PreTrainedTokenizerBase):
        if not self._is_word_level(tokenizer):
            raise ValueError(
                "WordLevelStopCriteria is only compatible with WordLevel tokenizers. "
                "Detected a sub-word or character-level tokenizer which may cause "
                "ID mismatches during generation."
            )

        self.stop_sequences = [torch.tensor(tokenizer.encode(s, add_special_tokens=False)) for s in stop_strings if s]
        self.finished_mask = None
        self.device_synced = False

    def _is_word_level(self, tokenizer: PreTrainedTokenizerBase) -> bool:
        """Checks if the underlying tokenizer model is WordLevel."""
        # Check the 'backend_tokenizer' which contains the actual model type
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            # Check for the specific WordLevel model class string or type
            model_type = str(type(backend.model)).lower()
            return "wordlevel" in model_type

        # Fallback: Some older tokenizers might not expose backend_tokenizer clearly
        # or might just be a wrapper. You can also check the class name of the tokenizer.
        return "WordLevel" in type(tokenizer).__name__

    def __call__(self, input_ids: torch.LongTensor, **kwargs: Any) -> bool:
        # Move tensors to device once to save bandwidth
        if not self.device_synced:
            self.stop_sequences = [s.to(input_ids.device) for s in self.stop_sequences]
            self.device_synced = True

        batch_size = input_ids.shape[0]
        if self.finished_mask is None or self.finished_mask.shape[0] != batch_size:
            self.finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for stop_seq in self.stop_sequences:
            seq_len = stop_seq.shape[0]
            if input_ids.shape[1] < seq_len:
                continue

            # Vectorized check across batch
            matches = (input_ids[:, -seq_len:] == stop_seq).all(dim=-1)
            self.finished_mask |= matches

        if self.finished_mask is None:
            return False

        return bool(self.finished_mask.all().item())
