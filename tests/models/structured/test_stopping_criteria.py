from types import SimpleNamespace

import pytest
import torch

from mimic_struct_predict.models.structured.stopping_criteria import TimeDeltaStoppingCriteria, WordLevelStopCriteria


class DummyTokenizer:
    def __init__(self, mapping=None, wordlevel=True, vocab=None):
        self.mapping = mapping or {}
        self._vocab = vocab or {}
        # Provide a backend_tokenizer.model whose type name contains 'WordLevel'
        if wordlevel:

            class WordLevelModel:
                pass

            self.backend_tokenizer = SimpleNamespace(model=WordLevelModel())

    def encode(self, s, add_special_tokens=False):
        # Return mapped ids if present, otherwise generate a deterministic small list
        if s in self.mapping:
            return list(self.mapping[s])
        return [ord(c) % 97 + 10 for c in s]

    def get_vocab(self):
        return dict(self._vocab)

    def __len__(self):
        ids = []
        if self._vocab:
            ids.extend(self._vocab.values())
        for seq in self.mapping.values():
            ids.extend(seq)
        return max(ids) + 1 if ids else 1


def _pad_seqs(seqs, pad=0):
    max_len = max(len(s) for s in seqs)
    return torch.tensor([[pad] * (max_len - len(s)) + s for s in seqs], dtype=torch.long)


def test_single_token_stop():
    tok = DummyTokenizer({"STOP": [9]})
    crit = WordLevelStopCriteria(["STOP"], tok)

    # no sequences have stop token yet
    inp = _pad_seqs([[1], [2], [3]])
    out = crit(inp)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.bool and out.shape == (3, 1)
    assert not out.all().item()

    # one sequence ends with STOP -> still False (not all finished)
    inp = _pad_seqs([[1, 9], [2], [3]])
    out = crit(inp)
    assert out.shape == (3, 1)
    assert out.squeeze(1).tolist() == [True, False, False]
    assert not out.all().item()

    # now all sequences end with STOP -> True
    inp = _pad_seqs([[1, 9], [2, 9], [3, 9]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True, True]
    assert out.all().item()


def test_multi_token_stop():
    tok = DummyTokenizer({"END": [5, 6]})
    crit = WordLevelStopCriteria(["END"], tok)

    # none have end sequence
    inp = _pad_seqs([[1], [1]])
    out = crit(inp)
    assert out.shape == (2, 1) and not out.all().item()

    # partial tokens present but not full match
    inp = _pad_seqs([[1, 5], [1]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [False, False]
    assert not out.all().item()

    # first sequence has full END match (but second hasn't) -> still False
    inp = _pad_seqs([[5, 6], [1]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, False]
    assert not out.all().item()

    # now second obtains END -> True
    inp = _pad_seqs([[5, 6], [9, 5, 6]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()


def test_empty_stop_list_returns_false():
    tok = DummyTokenizer()
    crit = WordLevelStopCriteria([], tok)
    inp = _pad_seqs([[1]])
    out = crit(inp)
    assert out.shape == (1, 1) and out.dtype == torch.bool
    assert not out.all().item()


def test_multiple_stop_words():
    tok = DummyTokenizer({"STOP": [9], "END": [5, 6]})
    crit = WordLevelStopCriteria(["STOP", "END"], tok)

    # none have stop tokens
    inp = _pad_seqs([[1], [2]])
    out = crit(inp)
    assert out.shape == (2, 1) and not out.all().item()

    # one has full STOP, other has partial END -> still False
    inp = _pad_seqs([[1, 9], [2, 5]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, False]
    assert not out.all().item()

    # now second has full END -> True (each sequence hit at least one stop word)
    inp = _pad_seqs([[1, 9], [2, 5, 6]])
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()


def test_non_word_level_tokenizer_raises():
    # DummyTokenizer with wordlevel=False should not appear as WordLevel
    tok = DummyTokenizer(wordlevel=False)
    with pytest.raises(ValueError):
        WordLevelStopCriteria(["STOP"], tok)


def test_time_delta_ignores_prompt_and_counts_generated_tokens_only():
    tok = DummyTokenizer(vocab={"+10 days": 10, "+2 days": 2})
    crit = TimeDeltaStoppingCriteria(tokenizer=tok, max_time_delta=3, prompt_length=2)

    # Prompt contains large delta tokens, but prompt portion must be ignored.
    inp = torch.tensor([[10, 10, 2], [10, 10, 2]], dtype=torch.long)
    out = crit(inp)
    assert out.shape == (2, 1) and out.dtype == torch.bool
    assert not out.all().item()

    # Newly generated tokens exceed the threshold for all sequences.
    inp = torch.tensor([[10, 10, 2, 2], [10, 10, 2, 2]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()


def test_time_delta_requires_all_sequences_to_exceed_limit():
    tok = DummyTokenizer(vocab={"+2 days": 2})
    crit = TimeDeltaStoppingCriteria(tokenizer=tok, max_time_delta=3, prompt_length=1)

    # First sequence exceeds threshold (4), second does not (2).
    inp = torch.tensor([[0, 2, 2], [0, 2, 1]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, False]
    assert not out.all().item()

    # Now both sequences exceed threshold.
    inp = torch.tensor([[0, 2, 2], [0, 2, 2]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()


def test_time_delta_uses_strict_greater_than_for_stop_condition():
    tok = DummyTokenizer(vocab={"+3 days": 3})
    crit = TimeDeltaStoppingCriteria(tokenizer=tok, max_time_delta=3, prompt_length=1)

    # Equal to threshold should not stop because criterion uses `>`.
    inp = torch.tensor([[0, 3], [0, 3]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [False, False]
    assert not out.all().item()

    inp = torch.tensor([[0, 3, 3], [0, 3, 3]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()


def test_time_delta_regex_extracts_day_tokens_and_ignores_others():
    tok = DummyTokenizer(vocab={"+1 day": 1, " +2 days": 2, "+5 hours": 5, "event": 7})
    crit = TimeDeltaStoppingCriteria(tokenizer=tok, max_time_delta=2, prompt_length=0)

    # Only day/day tokens should contribute: 1 + 2 = 3 (> 2).
    inp = torch.tensor([[1, 2, 5, 7], [1, 2, 5, 7]], dtype=torch.long)
    out = crit(inp)
    assert out.squeeze(1).tolist() == [True, True]
    assert out.all().item()
