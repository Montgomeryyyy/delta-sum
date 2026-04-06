import polars as pl
import pytest
from polars.testing import assert_frame_equal
from omegaconf import DictConfig
from mimic_struct_predict.preprocessing.meds.transforms.regex_replacement.regex_replacement import regex_replacement

def test_regex_replacement_applies_patterns():
    # Arrange
    df = pl.DataFrame({
        "code": ["prefix_HELLO_123", "prefix_WORLD_456"]
    }).lazy()
    cfg = DictConfig({
        "column": "code",
        "patterns": [
            {"pattern": "prefix_", "replacement": ""},
            {"pattern": "_[0-9]+", "replacement": ""},
        ]
    })
    fn = regex_replacement(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    assert result["code"].to_list() == ["HELLO", "WORLD"]

def test_regex_replacement_case_sensitivity():
    # Arrange
    df = pl.DataFrame({
        "code": ["Foo", "foo"]
    }).lazy()
    cfg = DictConfig({
        "column": "code",
        "patterns": [
            {"pattern": "^f", "replacement": "X"},
        ]
    })
    fn = regex_replacement(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # By default regex is case sensitive
    assert result["code"].to_list() == ["Foo", "Xoo"]
