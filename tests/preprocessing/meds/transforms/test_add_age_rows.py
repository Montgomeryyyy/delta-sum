from datetime import datetime

import polars as pl
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.add_age_rows.add_age_rows import add_age_rows


def test_add_age_rows_inserts_correct_ages():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2020, 1, 1),  # Exactly 20 years
                datetime(2020, 6, 15),  # 20 years and some days
            ],
            "code": ["BORN", "LAB", "DIAG"],
        }
    ).lazy()
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

    expected = pl.DataFrame(
        {
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
                "TEST//",
                "BORN",
                "TEST//",
                "LAB",
                "TEST//",
                "DIAG",
            ],
            "age_text_value": [
                "0 years//0 days",
                None,
                "20 years//0 days",
                None,
                "20 years//166 days",
                None,
            ],
            "numeric_value": [
                0.0,
                None,
                20.0,
                None,
                20.0 + 166.0 / 365.25,
                None,
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_add_age_rows_multiple_subjects():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "time": [
                datetime(2000, 1, 1),
                datetime(2010, 1, 1),
                datetime(1990, 1, 1),
                datetime(2000, 1, 1),
            ],
            "code": ["BORN", "EVENT", "BORN", "EVENT"],
        }
    ).lazy()
    cfg = DictConfig({"birth_code": "BORN"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["Age", "BORN", "Age", "EVENT", "Age", "BORN", "Age", "EVENT"]
    assert result["code"].to_list() == expected_codes
    assert result["age_text_value"][:2].to_list() == ["0 years//0 days", None]
    assert result["subject_id"].to_list() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_add_age_rows_leap_year_handling():
    """Verify correct age calculation across leap year boundaries."""
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 1, 1, 2, 2],
            "time": [
                datetime(2000, 2, 29),  # Born on Leap Day
                datetime(2001, 2, 27),  # Day before 1st anniversary
                datetime(2001, 2, 28),  # 1st anniversary (polars behavior for Feb 29 + 1y)
                datetime(2001, 3, 1),  # Day after 1st anniversary
                datetime(2004, 2, 29),  # 4th anniversary (Leap Year)
                datetime(1999, 2, 28),  # Subject 2: Born Feb 28
                datetime(2000, 2, 29),  # 1st anniversary + 1 day (Leap Year)
            ],
            "code": ["BORN", "E1", "E2", "E3", "E4", "BORN", "E1"],
        }
    ).lazy()
    cfg = DictConfig({"birth_code": "BORN"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # Subject 1 (Born 2000-02-29)
    sub1 = result.filter(pl.col("subject_id") == 1)

    # helper to get age text at a specific time
    def get_age(df, dt):
        return df.filter((pl.col("code") == "Age") & (pl.col("time") == dt)).select("age_text_value").item()

    # E1: 2001-02-27 -> 0 years, 364 days
    assert get_age(sub1, datetime(2001, 2, 27)) == "0 years//364 days"
    # E2: 2001-02-28 -> 1 years, 0 days
    assert get_age(sub1, datetime(2001, 2, 28)) == "1 years//0 days"
    # E3: 2001-03-01 -> 1 years, 1 days
    assert get_age(sub1, datetime(2001, 3, 1)) == "1 years//1 days"
    # E4: 2004-02-29 -> 4 years, 0 days
    assert get_age(sub1, datetime(2004, 2, 29)) == "4 years//0 days"

    # Subject 2 (Born 1999-02-28)
    sub2 = result.filter(pl.col("subject_id") == 2)
    # E1: 2000-02-29 -> 1 years, 1 day
    assert get_age(sub2, datetime(2000, 2, 29)) == "1 years//1 days"


def test_add_age_rows_leap_year_increment_days():
    """Verify that leap years correctly increment the day count irrespective of birthdate."""
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [
                datetime(1999, 1, 1),  # Sub 1: Born Jan 1
                datetime(2000, 3, 1),  # Leap year check (2000)
                datetime(1998, 1, 1),  # Sub 2: Born Jan 1
                datetime(1999, 3, 1),  # Non-leap year check (1999)
                datetime(1999, 3, 1),  # Sub 3: Born Mar 1
                datetime(2000, 2, 29), # Leap year check (2000)
                datetime(1997, 3, 1),  # Sub 4: Born Mar 1
                datetime(1998, 2, 28), # Non-leap year check (1998)
            ],
            "code": ["BORN", "E1", "BORN", "E1", "BORN", "E1", "BORN", "E1"],
        }
    ).lazy()
    cfg = DictConfig({"birth_code": "BORN"})
    fn = add_age_rows(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    def get_age(df, sid, dt):
        return (
            df.filter((pl.col("subject_id") == sid) & (pl.col("code") == "Age") & (pl.col("time") == dt))
            .select("age_text_value")
            .item()
        )

    # Subject 1: Born 1999-01-01, checking 2000-03-01 (Leap Year)
    # In leap year 2000, Jan 1 to Mar 1 is 31 (Jan) + 29 (Feb) = 60 days
    assert get_age(result, 1, datetime(2000, 3, 1)) == "1 years//60 days"

    # Subject 2: Born 1998-01-01, checking 1999-03-01 (Non-Leap Year)
    # In non-leap year 1999, Jan 1 to Mar 1 is 31 (Jan) + 28 (Feb) = 59 days
    assert get_age(result, 2, datetime(1999, 3, 1)) == "1 years//59 days"

    # Subject 3: Born 1999-03-01, checking 2000-02-29 (Leap Year)
    # Since it's a leap year, the day before the birthday is day 365
    assert get_age(result, 3, datetime(2000, 2, 29)) == "0 years//365 days"

    # Subject 4: Born 1997-03-01, checking 1998-02-28 (Non-Leap Year)
    # In a non-leap year, the day before the birthday is day 364
    assert get_age(result, 4, datetime(1998, 2, 28)) == "0 years//364 days"
