import polars as pl
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.merge_age_column.merge_age_column import (
    merge_age_column,
)


def test_merge_age_column_basic_merges_and_nulls():
    """Verify basic functionality with age_text_value: merge code and age_text_value, null numeric_value."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["Age", "LAB", "Age"],
            "age_text_value": ["20 years", None, "30 years"],
            "numeric_value": [20.0, 1.2, 30.0],
        }
    ).lazy()
    # Explicitly set drop_age_text_column to False to verify merging without dropping
    cfg = DictConfig({"drop_age_text_column": False})
    fn = merge_age_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["Age//20 years", "LAB", "Age//30 years"],
            "age_text_value": ["20 years", None, "30 years"],
            "numeric_value": [None, 1.2, None],
        }
    )
    assert_frame_equal(result, expected)


def test_merge_age_column_custom_age_code():
    """Verify that the transform respects a custom age_code in config."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["Patient Age", "LAB"],
            "age_text_value": ["45 years", "Normal"],
            "numeric_value": [45.0, 1.0],
        }
    ).lazy()
    cfg = DictConfig({"age_code": "Patient Age", "drop_age_text_column": False})
    fn = merge_age_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["Patient Age//45 years", "LAB"],
            "age_text_value": ["45 years", "Normal"],
            "numeric_value": [None, 1.0],
        }
    )
    assert_frame_equal(result, expected)


def test_merge_age_column_handles_null_text_value():
    """Verify that numeric_value is nulled even if age_text_value is null (no merge)."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["Age", "Age"],
            "age_text_value": [None, "50 years"],
            "numeric_value": [0.0, 50.0],
        }
    ).lazy()
    cfg = DictConfig({"drop_age_text_column": False})
    fn = merge_age_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["Age", "Age//50 years"],
            "age_text_value": [None, "50 years"],
            "numeric_value": [None, None],
        },
        schema={
            "code": pl.String,
            "age_text_value": pl.String,
            "numeric_value": pl.Float64,
        },
    )
    assert_frame_equal(result, expected)


def test_merge_age_column_default_drops_column():
    """Verify that age_text_value column is dropped by default."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["Age", "LAB"],
            "age_text_value": ["20 years", None],
            "numeric_value": [20.0, 1.2],
        }
    ).lazy()
    cfg = DictConfig({})  # Default should drop column
    fn = merge_age_column(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["Age//20 years", "LAB"],
            "numeric_value": [None, 1.2],
        }
    )
    assert_frame_equal(result, expected)
    assert "age_text_value" not in result.columns
