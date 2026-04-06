import polars as pl
import pytest
from polars.testing import assert_frame_equal
from omegaconf import DictConfig
from mimic_struct_predict.preprocessing.meds.transforms.order_by_code_prefix.order_by_code_prefix import order_by_code_prefix

def test_order_by_code_prefix_sorts_correctly():
    # Arrange
    df = pl.DataFrame({
        "subject_id": [1, 1, 1, 1],
        "time": [1, 1, 1, 1],
        "code": [
            "MED//aspirin",
            "LAB//glucose",
            "ADMISSION//emergency",
            "UNKNOWN//thing",
        ],
        "numeric_value": [None, 1.0, None, None]
    }).lazy()
    cfg = DictConfig({"prefix_priority": ["ADMISSION", "LAB", "MED"]})
    fn = order_by_code_prefix(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = [
        "ADMISSION//emergency",
        "LAB//glucose",
        "MED//aspirin",
        "UNKNOWN//thing",
    ]
    assert result["code"].to_list() == expected_codes

def test_order_by_code_prefix_secondary_sorts():
    # Arrange
    df = pl.DataFrame({
        "subject_id": [1, 1, 1],
        "time": [1, 1, 1],
        "code": ["LAB//B", "LAB//A", "LAB//A"],
        "numeric_value": [None, 2.0, 1.0]
    }).lazy()
    cfg = DictConfig({"prefix_priority": ["LAB"]})
    fn = order_by_code_prefix(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # 1. LAB//A (numeric 1.0)
    # 2. LAB//A (numeric 2.0)
    # 3. LAB//B (numeric None)
    expected_codes = ["LAB//A", "LAB//A", "LAB//B"]
    expected_values = [1.0, 2.0, None]
    assert result["code"].to_list() == expected_codes
    assert result["numeric_value"].to_list() == expected_values

def test_order_by_code_prefix_empty_priority():
    # Arrange
    df = pl.DataFrame({
        "subject_id": [1, 1],
        "time": [1, 1],
        "code": ["B", "A"],
        "numeric_value": [None, None]
    }).lazy()
    cfg = DictConfig({"prefix_priority": []})
    fn = order_by_code_prefix(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # Should fallback to sorting by code (since priority is same for all)
    assert result["code"].to_list() == ["A", "B"]
