from datetime import datetime
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from omegaconf import DictConfig
from mimic_struct_predict.preprocessing.meds.transforms.add_age_rows.add_age_rows import add_age_rows

def test_add_age_rows_inserts_correct_ages():
    # Arrange
    df = pl.DataFrame({
        "subject_id": [1, 1, 1],
        "time": [
            datetime(2000, 1, 1),
            datetime(2020, 1, 1), # Exactly 20 years
            datetime(2020, 6, 15), # 20 years and some days
        ],
        "code": ["BORN", "LAB", "DIAG"],
    }).lazy()
    cfg = DictConfig({"birth_code": "BORN", "header": "TEST//"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # Expected rows:
    # 1. Age 0 years//0 days (Parent for row 0)
    # 2. BORN
    # 3. Age 20 years//0 days (Parent for row 1)
    # 4. LAB
    # 5. Age 20 years//166 days (Parent for row 2)
    # 6. DIAG
    
    expected = pl.DataFrame({
        "subject_id": [1, 1, 1, 1, 1, 1],
        "time": [
            datetime(2000, 1, 1),
            datetime(2000, 1, 1),
            datetime(2020, 1, 1),
            datetime(2020, 1, 1),
            datetime(2020, 6, 15),
            datetime(2020, 6, 15),
        ],
        "code": [
            "TEST//Age 0 years//0 days",
            "BORN",
            "TEST//Age 20 years//0 days",
            "LAB",
            "TEST//Age 20 years//166 days",
            "DIAG",
        ],
    })
    assert_frame_equal(result, expected)

def test_add_age_rows_multiple_subjects():
    # Arrange
    df = pl.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "time": [
            datetime(2000, 1, 1),
            datetime(2010, 1, 1),
            datetime(1990, 1, 1),
            datetime(2000, 1, 1),
        ],
        "code": ["BORN", "EVENT", "BORN", "EVENT"],
    }).lazy()
    cfg = DictConfig({"birth_code": "BORN"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = [
        "Age 0 years//0 days", "BORN",
        "Age 10 years//0 days", "EVENT",
        "Age 0 years//0 days", "BORN",
        "Age 10 years//0 days", "EVENT",
    ]
    assert result["code"].to_list() == expected_codes
    assert result["subject_id"].to_list() == [1, 1, 1, 1, 2, 2, 2, 2]

def test_add_age_rows_handles_leap_years_roughly():
    # Arrange
    # 1999-01-01 to 2001-01-01 is 2 years (no leap day in between? wait 2000 is leap)
    # 2000-01-01 to 2020-01-01 is 20 years
    df = pl.DataFrame({
        "subject_id": [1, 1],
        "time": [
            datetime(2000, 2, 28),
            datetime(2004, 2, 28), # Exactly 4 years
        ],
        "code": ["BORN", "EVENT"],
    }).lazy()
    cfg = DictConfig({"birth_code": "BORN"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    assert result["code"][2] == "Age 4 years//0 days"
