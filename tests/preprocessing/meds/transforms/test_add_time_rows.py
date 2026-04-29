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
    cfg = DictConfig({"header": "T"})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # 1. T// (Parent for row 0)
    # 2. A
    # 3. B
    # 4. T// (Parent for row 2)
    # 5. C
    expected_codes = ["T//10h//00m", "A", "B", "T//10h//05m", "C"]
    assert result["code"].to_list() == expected_codes
    assert result.shape[0] == 5


def test_add_time_rows_ignores_seconds_and_micros_within_minute():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2024, 1, 1, 10, 0, 0, 0),
                datetime(2024, 1, 1, 10, 0, 59, 999999),  # Same minute, different seconds/micros
                datetime(2024, 1, 1, 10, 1, 0, 0),
            ],
            "code": ["A", "B", "C"],
        }
    ).lazy()
    cfg = DictConfig({"header": "T"})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["T//10h//00m", "A", "B", "T//10h//01m", "C"]
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
    expected_codes = ["10h//00m", "A", "10h//00m", "B"]
    assert result["code"].to_list() == expected_codes
    assert result["subject_id"].to_list() == [1, 1, 2, 2]


def test_add_time_rows_date_change_same_minute_inserts():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [
                datetime(2024, 1, 1, 10, 0, 30),
                datetime(2024, 1, 2, 10, 0, 0),  # Different day, same HH:MM
            ],
            "code": ["A", "B"],
        }
    ).lazy()
    cfg = DictConfig({"header": "T"})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # A new parent row should be inserted when the timestamp changes at minute precision,
    # even if the HH:MM string is identical.
    expected_codes = ["T//10h//00m", "A", "T//10h//00m", "B"]
    assert result["code"].to_list() == expected_codes
    assert result.shape[0] == 4


def test_add_time_rows_eod():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 23, 59, 10),  # Natural event in the last minute
                datetime(2024, 1, 1, 23, 59, 59, 999999),  # Artificial End of day
            ],
            "code": ["A", "B", "C"],
        }
    ).lazy()
    cfg = DictConfig({"header": "T", "eod_string": "END_OF_DAY"})
    fn = add_time_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["T//10h//00m", "A", "T//23h//59m", "B", "T//END_OF_DAY", "C"]
    assert result["code"].to_list() == expected_codes
    assert result.shape[0] == 6
