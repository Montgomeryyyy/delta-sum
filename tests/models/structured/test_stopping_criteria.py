from types import SimpleNamespace

import pytest
import torch

from mimic_struct_predict.models.structured.stopping_criteria import WordLevelStopCriteria


class DummyTokenizer:
    def __init__(self, mapping=None, wordlevel=True):
        self.mapping = mapping or {}
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


def _pad_seqs(seqs, pad=0):
    max_len = max(len(s) for s in seqs)
    return torch.tensor([[pad] * (max_len - len(s)) + s for s in seqs], dtype=torch.long)


def test_single_token_stop():
    tok = DummyTokenizer({"STOP": [9]})
    crit = WordLevelStopCriteria(["STOP"], tok)

    # no sequences have stop token yet
    inp = _pad_seqs([[1], [2], [3]])
    assert crit(inp) is False

    # one sequence ends with STOP -> still False (not all finished)
    inp = _pad_seqs([[1, 9], [2], [3]])
    assert crit(inp) is False

    # now all sequences end with STOP -> True
    inp = _pad_seqs([[1, 9], [2, 9], [3, 9]])
    assert crit(inp) is True


def test_multi_token_stop():
    tok = DummyTokenizer({"END": [5, 6]})
    crit = WordLevelStopCriteria(["END"], tok)

    # none have end sequence
    inp = _pad_seqs([[1], [1]])
    assert crit(inp) is False

    # partial tokens present but not full match
    inp = _pad_seqs([[1, 5], [1]])
    assert crit(inp) is False

    # first sequence has full END match (but second hasn't) -> still False
    inp = _pad_seqs([[5, 6], [1]])
    assert crit(inp) is False

    # now second obtains END -> True
    inp = _pad_seqs([[5, 6], [9, 5, 6]])
    assert crit(inp) is True


def test_empty_stop_list_returns_false():
    tok = DummyTokenizer()
    crit = WordLevelStopCriteria([], tok)
    inp = _pad_seqs([[1]])
    assert crit(inp) is False


def test_multiple_stop_words():
    tok = DummyTokenizer({"STOP": [9], "END": [5, 6]})
    crit = WordLevelStopCriteria(["STOP", "END"], tok)

    # none have stop tokens
    inp = _pad_seqs([[1], [2]])
    assert crit(inp) is False

    # one has full STOP, other has partial END -> still False
    inp = _pad_seqs([[1, 9], [2, 5]])
    assert crit(inp) is False

    # now second has full END -> True (each sequence hit at least one stop word)
    inp = _pad_seqs([[1, 9], [2, 5, 6]])
    assert crit(inp) is True


def test_non_word_level_tokenizer_raises():
    # DummyTokenizer with wordlevel=False should not appear as WordLevel
    tok = DummyTokenizer(wordlevel=False)
    with pytest.raises(ValueError):
        WordLevelStopCriteria(["STOP"], tok)
