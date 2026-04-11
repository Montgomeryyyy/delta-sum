from datetime import datetime

import polars as pl
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.add_delta_rows.add_delta_rows import add_delta_rows


def test_add_delta_rows_inserts_delta_rows_on_subject_or_date_change():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1],
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 12, 0),  # same date -> no new delta row
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
            ],
            "code": ["A", "B", "C", "D"],
            "numeric_value": [None, None, None, 12.0],
        }
    ).lazy()
    cfg = DictConfig({"header": "Delta", "max_delta_days": 364})
    fn = add_delta_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 1],
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 12, 0),
                datetime(2020, 1, 3),
                datetime(2020, 1, 3),
                datetime(2020, 1, 4),
                datetime(2020, 1, 4),
            ],
            "code": [
                "A",
                "B",
                "Delta +2 days",
                "C",
                "Delta +1 days",
                "D",
            ],
            "numeric_value": [
                None,
                None,
                None,
                None,
                None,
                12.0,
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_add_delta_rows_splits_large_gap_by_max_delta_days():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2001, 1, 5),  # 370 day gap
            ],
            "code": ["A", "B"],
        }
    ).lazy()
    cfg = DictConfig({"header": "Delta", "max_delta_days": 364})
    fn = add_delta_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["A", "Delta +364 days", "Delta +6 days", "B"]
    assert result["code"].to_list() == expected_codes


def test_add_delta_rows_resets_delta_per_subject():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 10),
                datetime(2020, 1, 10),
                datetime(2020, 1, 12),
            ],
            "code": ["A", "B", "C", "D"],
        }
    ).lazy()
    cfg = DictConfig({"header": "Delta", "max_delta_days": 364})
    fn = add_delta_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = [
        "A",
        "Delta +9 days",
        "B",
        "C",
        "Delta +2 days",
        "D",
    ]
    assert result["code"].to_list() == expected_codes


def test_add_delta_rows_uses_custom_max_delta_days():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 11)],  # 10 day gap
            "code": ["A", "B"],
        }
    ).lazy()
    cfg = DictConfig({"header": "Delta", "max_delta_days": 3})
    fn = add_delta_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["A", "Delta +3 days", "Delta +3 days", "Delta +3 days", "Delta +1 days", "B"]
    assert result["code"].to_list() == expected_codes
