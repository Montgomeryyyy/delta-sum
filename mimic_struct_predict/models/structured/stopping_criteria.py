import re
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

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor | None = None, **kwargs: Any
    ) -> torch.BoolTensor:
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

        # Transformers expects a torch.BoolTensor with shape (batch_size)
        return self.finished_mask


class TimeDeltaStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that halts text generation when
    the cumulative time delta of the newly generated tokens exceeds a specified limit.

    This criteria is optimized for GPU execution using a dense lookup tensor ($O(1)$
    lookup per token) rather than iterating through dictionaries.

    To maximize efficiency and maintain compatibility with complex generation
    strategies. It then ignores the prompt entirely on passes, summing only
    the time deltas of the newly generated tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_time_delta: int, prompt_length: int):
        """
        Initializes the TimeDeltaStoppingCriteria.
        Must be re-initialized for each generation run to reset internal state.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer used for the model.
                Used to extract the vocabulary and identify time delta tokens.
                Assumed time delta format is "+N day(s)" where N is a positive integer.
                Assumes that the tokenizer does not split the time delta tokens into sub-tokens,
                which is true for specially designed WordLevel tokenizers.
            max_time_delta (int): The maximum cumulative time delta (in days)
                allowed for the newly generated text. Once exceeded, generation stops.
            prompt_length (int): The length of the initial prompt in tokens. This is used to
                ensure that only newly generated tokens are considered for the time delta sum.

        Note: We are only counting day time deltas not hours or minutes for simplicity,
        but this can be extended if needed.
        """
        self.max_time_delta = max_time_delta
        self.device_synced = False

        # State variables for generation tracking
        self.prompt_length = prompt_length

        # Initialize a zero-tensor for the entire vocabulary
        vocab_size = len(tokenizer)
        self.delta_lookup = torch.zeros(vocab_size, dtype=torch.int32)

        # Extract time delta mappings from the vocabulary using regex.
        # This matches tokens like '+5 days', ' +1 day', 'Ġ+10 days', etc.,
        # handling special prefix characters added by various tokenizers.
        pattern = re.compile(r"\+(\d+)\s*days?")

        for token, token_id in tokenizer.get_vocab().items():
            match = pattern.search(token)
            if match:
                self.delta_lookup[token_id] = int(match.group(1))

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor | None = None, **kwargs: Any
    ) -> torch.BoolTensor:
        """
        Evaluates whether the generation should stop based on the cumulative
        time deltas of the generated portion of the sequence.

        Args:
            input_ids (torch.LongTensor): The sequence of token IDs generated so far,
                including the initial prompt. Shape: (batch_size, sequence_length).
            scores (torch.FloatTensor): The logits for the next token. Shape: (batch_size, vocab_size).
            **kwargs: Additional keyword arguments passed by the generation loop.

        Returns:
            bool: True if all sequences in the batch have exceeded
                the max_time_delta limit, False otherwise.
        """
        # 1. Sync the lookup tensor to the active device (e.g., GPU) on the first step
        if not self.device_synced:
            self.delta_lookup = self.delta_lookup.to(input_ids.device)
            self.device_synced = True

        # 2. Isolate the newly generated tokens, ignoring the massive prompt context.
        # This ensures we don't redundantly sum the input prompt at every generation step.
        generated_ids = input_ids[:, self.prompt_length :]

        # 3. Vectorized lookup and sum: map all generated token IDs to their delta
        # values and sum them along the sequence dimension.
        # Shape: (batch_size, generated_seq_len) -> (batch_size,)
        generated_deltas = self.delta_lookup[generated_ids].sum(dim=-1)

        # 4. Evaluate which sequences in the batch have exceeded the threshold.
        is_done = generated_deltas > self.max_time_delta

        # Return a torch.BoolTensor of shape (batch_size) per docs
        return is_done
