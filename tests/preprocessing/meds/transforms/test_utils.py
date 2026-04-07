import polars as pl
from polars.testing import assert_frame_equal

from mimic_struct_predict.preprocessing.meds.transforms.utils import _changed, insert_parent_rows


def test__changed_detects_changes():
    # Arrange
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
        }
    ).lazy()

    # Act
    result = df.with_columns(changed=_changed("a")).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "changed": [True, False, True, False, True],
        }
    )
    assert_frame_equal(result, expected)


def test__changed_with_custom_fill_null():
    # Arrange
    df = pl.DataFrame(
        {
            "a": [1, 1, 2],
        }
    ).lazy()

    # Act
    result = df.with_columns(changed=_changed("a", fill_null=False)).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "changed": [False, False, True],
        }
    )
    assert_frame_equal(result, expected)


def test_insert_parent_rows_inserts_correctly():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "time": [10, 10, 20],
            "code": ["a", "b", "c"],
            "other": [1, 2, 3],
        }
    ).lazy()
    change_cond = _changed("subject_id")
    code_expr = pl.lit("PARENT")

    # Act
    result = insert_parent_rows(
        df,
        change_cond=change_cond,
        code_expr=code_expr,
    ).collect()

    # Assert
    expected = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [10, 10, 10, 20, 20],
            "code": ["PARENT", "a", "b", "PARENT", "c"],
            "other": [None, 1, 2, None, 3],
        },
        schema_overrides={"other": pl.Int64},
    )
    assert_frame_equal(result, expected)


def test_insert_parent_rows_preserves_all_columns():
    # Arrange
    df = pl.DataFrame(
        {
            "subject_id": [1],
            "time": [10],
            "code": ["a"],
            "keep_val": [100],
            "drop_me": [200],
        }
    ).lazy()
    # change_cond is always True to trigger parent row insertion
    change_cond = pl.lit(True)
    code_expr = pl.lit("PARENT")

    # Act
    # Without exclude_cols, all columns are preserved
    result = insert_parent_rows(
        df,
        change_cond=change_cond,
        code_expr=code_expr,
    ).collect()

    # Assert
    # We expect 'drop_me' and 'keep_val' to be nulled in the parent row
    expected = pl.DataFrame(
        {
            "subject_id": [1, 1],
            "time": [10, 10],
            "code": ["PARENT", "a"],
            "keep_val": [None, 100],
            "drop_me": [None, 200],
        },
        schema_overrides={"keep_val": pl.Int64, "drop_me": pl.Int64},
    )
    assert_frame_equal(result, expected)
