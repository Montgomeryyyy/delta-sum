import polars as pl
import pytest
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.replace_quantile_with_numeric_value.replace_quantile_with_numeric_value import (
    replace_quantile_with_numeric_value,
)


def test_replace_quantile_with_numeric_value_default_config_converts_correctly():
    """Verify conversion of various numeric values to code suffixes with default sig_figs=3."""
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [1, 2, 3],
            "code": ["HR", "BP//SYS", "SPO2"],
            "numeric_value": [72.0, 120.56, 98.5],
            "text_value": [None, None, "Existing"],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "time": [1, 2, 3],
            "code": ["HR//72", "BP//121.0", "SPO2//98.5"],
            "numeric_value": [72.0, 120.56, 98.5],
            "text_value": [None, None, "Existing"],
        },
        schema={
            "subject_id": pl.Int64,
            "time": pl.Int64,
            "code": pl.String,
            "numeric_value": pl.Float64,
            "text_value": pl.String,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_drop_numeric_values_true_drops_column():
    """Verify that drop_numeric_values option drops the numeric_value column entirely."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["HR//Quantile 0.5", "BP//SYS//0.1", "TEMP"],
            "numeric_value": [72.0, 120.5, None],
        }
    ).lazy()
    cfg = DictConfig({"drop_numeric_values": True})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["HR//72", "BP//SYS//121.0", "TEMP"],
        },
        schema={
            "code": pl.String,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_preserves_other_columns_leaves_intact():
    """Verify that the transform doesn't affect other unrelated columns."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["A", "B"],
            "numeric_value": [1.0, 2.5],
            "text_value": [None, "T"],
            "extra": ["X", "Y"],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["A//1", "B//2.5"],
            "numeric_value": [1.0, 2.5],
            "text_value": [None, "T"],
            "extra": ["X", "Y"],
        },
        schema={
            "code": pl.String,
            "numeric_value": pl.Float64,
            "text_value": pl.String,
            "extra": pl.String,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_custom_sig_figs_rounds_correctly():
    """Verify configuration of sig_figs customizes rounding."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["HR", "BP", "TEMP"],
            "numeric_value": [72.1234, 120.5678, 37.0],
        }
    ).lazy()
    cfg = DictConfig({"sig_figs": 5})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["HR//72.123", "BP//120.57", "TEMP//37"],
            "numeric_value": [72.1234, 120.5678, 37.0],
        },
        schema={
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_multiple_delimiters_replaces_only_last_segment():
    """Verify that only the final segment matching // is replaced."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["A//B//C", "X//Y//Z//1.0"],
            "numeric_value": [10.0, 20.0],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["A//B//10", "X//Y//Z//20"],
            "numeric_value": [10.0, 20.0],
        },
        schema={
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_zero_and_negative_values_converts_correctly():
    """Verify conversion properly handles zeroes and negative values."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["A", "B", "C"],
            "numeric_value": [0.0, -15.2, -0.005],
        }
    ).lazy()
    cfg = DictConfig({"sig_figs": 2})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["A//0", "B//-15.0", "C//-0.005"],
            "numeric_value": [0.0, -15.2, -0.005],
        },
        schema={
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_no_delimiters_appends_value():
    """Verify that when no delimiter is present, the value is just appended with the delimiter."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["CODE1", "CODE2"],
            "numeric_value": [1.0, 2.0],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "code": ["CODE1//1", "CODE2//2"],
            "numeric_value": [1.0, 2.0],
        },
        schema={
            "code": pl.String,
            "numeric_value": pl.Float64,
        },
    )
    assert_frame_equal(result, expected)


def test_replace_quantile_with_numeric_value_missing_numeric_value_column_raises_error():
    """Verify that missing numeric_value column will raise Python/Polars error."""
    # Arrange
    df = pl.DataFrame(
        {
            "code": ["HR", "BP"],
        }
    ).lazy()
    cfg = DictConfig({})
    fn = replace_quantile_with_numeric_value(cfg)

    # Act & Assert
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        fn(df).collect()
