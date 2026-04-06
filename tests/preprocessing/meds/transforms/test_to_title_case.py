import polars as pl
import pytest
from polars.testing import assert_frame_equal
from omegaconf import DictConfig
from mimic_struct_predict.preprocessing.meds.transforms.to_title_case.title_case_codes import title_case_codes

def test_title_case_codes_works():
    # Arrange
    df = pl.DataFrame({
        "code": ["HELLO WORLD", "foo bar", "Already Title Case"]
    }).lazy()
    cfg = DictConfig({})
    fn = title_case_codes(cfg)

    # Act
    result = fn(df).collect()

    # Assert
    expected_codes = ["Hello World", "Foo Bar", "Already Title Case"]
    assert result["code"].to_list() == expected_codes
