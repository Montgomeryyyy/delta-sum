from datetime import datetime

import polars as pl
from omegaconf import DictConfig

from mimic_struct_predict.preprocessing.meds.transforms.add_time_rows.add_time_rows import add_time_rows


def test_add_time_rows_inserts_correctly():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 30),  # Same minute
                datetime(2024, 1, 1, 10, 5, 0),  # New minute
            ],
            "code": ["A", "B", "C"],
        }
    ).lazy()
    cfg = DictConfig({"header": "T//"})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # 1. T// (Parent for row 0)
    # 2. A
    # 3. B
    # 4. T// (Parent for row 2)
    # 5. C
    expected_codes = ["T//10 hours//00 mins", "A", "B", "T//10 hours//05 mins", "C"]
    assert result["code"].to_list() == expected_codes
    assert result.shape[0] == 5


def test_add_time_rows_multiple_subjects():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 2],
            "time": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 0),
            ],
            "code": ["A", "B"],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # Even if time is same, subject changed so new parent row
    expected_codes = ["Time//10 hours//00 mins", "A", "Time//10 hours//00 mins", "B"]
    assert result["code"].to_list() == expected_codes
    assert result["subject_id"].to_list() == [1, 1, 2, 2]
