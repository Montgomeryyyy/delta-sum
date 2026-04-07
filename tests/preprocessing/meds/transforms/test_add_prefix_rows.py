import polars as pl
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.add_prefix_rows.add_prefix_rows import create_prefix_column


def test_create_prefix_column_inserts_rows():
    # Arrange
    df = pl.DataFrame(
        {"subject_id": [1, 1, 2], "time": [1, 1, 3], "code": ["LAB//glucose", "LAB//sodium", "MEDICATION//aspirin"]}
    ).lazy()
    cfg = DictConfig({"header": "PREFIX//"})
    fn = create_prefix_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [1, 1, 1, 3, 3],
            "code": ["PREFIX//LAB", "glucose", "sodium", "PREFIX//MEDICATION", "aspirin"],
        }
    )
    assert_frame_equal(result, expected)


def test_create_prefix_column_no_delimiter():
    # Arrange
    df = pl.DataFrame({"subject_id": [1], "time": [1], "code": ["normal_code"]}).lazy()
    cfg = DictConfig({})
    fn = create_prefix_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # No prefix row should be inserted because prefix will be null
    expected = pl.DataFrame({"subject_id": [1], "time": [1], "code": ["normal_code"]})
    assert_frame_equal(result, expected)


def test_create_prefix_column_triggers_on_time_change():
    # Arrange
    df = pl.DataFrame({"subject_id": [1, 1], "time": [1, 2], "code": ["LAB//glucose", "LAB//sodium"]}).lazy()
    cfg = DictConfig({})
    fn = create_prefix_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    # even though prefix is same (LAB), time changed so new parent row
    expected_codes = ["LAB", "glucose", "LAB", "sodium"]
    assert result["code"].to_list() == expected_codes
