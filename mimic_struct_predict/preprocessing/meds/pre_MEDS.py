#!/usr/bin/env python
# Modified from https://github.com/ipolharvard/ethos-ares/blob/2d54383997318eb52f3d47b5969a66fc166b71ff/scripts/meds/mimic/pre_MEDS.py
#
# Changes from pre_MEDS_original.py:
# - Refactored main to allow for processing which depends on multiple tables and simplified the main function.
# - Added greater preprocessing and joins of tables. This is documented in the
#   docstring of each processing function.
# - Removed ICD dot preprocessing as descriptions not codes used downstream.
# - Small bug fixes (e.g. correctly identifying .csv.gz files)


"""Performs pre-MEDS data wrangling for MIMIC-IV."""

import logging
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, cast

import hydra
import polars as pl
from omegaconf import DictConfig

from mimic_struct_predict.preprocessing.meds.utils.extract_utils import get_supported_fp
from mimic_struct_predict.preprocessing.meds.utils.utils import get_shard_prefix, write_lazyframe

logger = logging.getLogger(__name__)

END_OF_DAY = pl.time(23, 59, 59, 999999)

### Helper Functions ###


def _convert_date_to_end_of_day_timestamp(df: pl.LazyFrame, date_column: str = "chartdate") -> pl.LazyFrame:
    """Converts a date-only column to a timestamp set at the end of the day."""
    return df.with_columns(pl.col(date_column).str.to_date().dt.combine(END_OF_DAY).alias(date_column))


def _join_with_items_lookup(
    events_df: pl.LazyFrame, d_items_df: pl.LazyFrame, columns_to_add: list[str]
) -> pl.LazyFrame:
    """Joins events with an items lookup table (e.g. `d_labitems`, `d_items`) by `itemid`."""
    d_items_df = d_items_df.select("itemid", *columns_to_add)
    return events_df.join(d_items_df, on="itemid", how="left")


def _set_value_empty_string_when_valuenum_present(events_df: pl.LazyFrame) -> pl.LazyFrame:
    """Sets `value` to empty string when `valuenum` is not null."""
    return events_df.with_columns(
        pl.when(pl.col("valuenum").is_not_null()).then(pl.lit("")).otherwise(pl.col("value")).alias("value")
    )


def _add_discharge_time_by_hadm_id(
    df: pl.LazyFrame, discharge_time_df: pl.LazyFrame, out_column_name: str = "hadm_discharge_time"
) -> pl.LazyFrame:
    """Joins the two dataframes by ``"hadm_id"`` and adds the discharge time to the original dataframe."""

    discharge_time_df = discharge_time_df.select("hadm_id", pl.col("dischtime").alias(out_column_name))
    return df.join(discharge_time_df, on="hadm_id", how="left")


### Process Functions ###


def process_admissions_df(admissions_df: pl.LazyFrame, patients_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins admissions with `patients` to add the standardised `gender` column."""
    patients_df = patients_df.select("subject_id", "gender").with_columns(
        pl.col("gender").replace_strict({"F": "Female", "M": "Male"}, default=pl.col("gender")),
    )

    return admissions_df.join(patients_df, on="subject_id", how="left")


def process_icd_df(icd_df: pl.LazyFrame, admissions_df: pl.LazyFrame, d_icd_df: pl.LazyFrame) -> pl.LazyFrame:
    """Adds discharge time from admissions, joins with `d_icd` for long titles,
    ensures codes are strings, and converts `chartdate` to end-of-day timestamps.
    """
    icd_df = _add_discharge_time_by_hadm_id(icd_df, admissions_df)
    # Cast to string to match d_icd schema
    icd_df = icd_df.with_columns(
        pl.col("icd_code").cast(pl.String),
        pl.col("icd_version").cast(pl.String),
    )
    icd_df = icd_df.join(
        d_icd_df.select("icd_code", "icd_version", "long_title"),
        on=["icd_code", "icd_version"],
        how="left",
    )
    # Convert chartdate to timestamp at 23:59:59 (applies to procedures_icd)
    if "chartdate" in icd_df.collect_schema():
        icd_df = _convert_date_to_end_of_day_timestamp(icd_df)
    return icd_df


def process_drgcodes_df(drgcodes_df: pl.LazyFrame, admissions_df: pl.LazyFrame) -> pl.LazyFrame:
    """Maps drg_severity and drg_mortality to free text categorical values
    and adds discharge time from admissions.
    """
    severity_map = {"1": "MINOR", "2": "MODERATE", "3": "MAJOR", "4": "EXTREME"}
    return drgcodes_df.with_columns(
        pl.col("drg_severity").replace_strict(severity_map, default=None).alias("drg_severity"),
        pl.col("drg_mortality").replace_strict(severity_map, default=None).alias("drg_mortality"),
    ).pipe(_add_discharge_time_by_hadm_id, admissions_df)


def process_hcpcs_events_df(hcpcs_events_df: pl.LazyFrame, d_hcpcs_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins with `d_hcpcs` to add procedure descriptions (preferring long,
    falling back to short) and converts `chartdate` to end-of-day timestamps.
    """
    d_hcpcs_df = d_hcpcs_df.select(
        "code",
        pl.coalesce("long_description", "short_description").alias("long_description"),
    )
    hcpcs_events_df = hcpcs_events_df.join(d_hcpcs_df, left_on="hcpcs_cd", right_on="code", how="left")
    return _convert_date_to_end_of_day_timestamp(hcpcs_events_df)


def process_lab_events_df(lab_events_df: pl.LazyFrame, d_labitems_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins lab events with `d_labitems` to add item labels and fluid types, sets null valueuom to empty string,
    and sets `value` to empty string when `valuenum` is not null.
    """
    return (
        _join_with_items_lookup(lab_events_df, d_labitems_df, ["fluid", "label"])
        .with_columns(pl.col("valueuom").fill_null(""))
        .pipe(_set_value_empty_string_when_valuenum_present)
    )


def process_omr_df(omr_df: pl.LazyFrame) -> pl.LazyFrame:
    """Extract numeric blood pressure, weight, height and BMI data, splitting
    blood pressure into systolic/diastolic rows.
    """
    output_cols = ["subject_id", "chartdate", "seq_num", "result_name", "result_num"]

    # Blood pressure: split on "/" and unpivot into separate rows
    bp_df = (
        omr_df.filter(
            pl.col("result_name").str.contains("Blood Pressure", literal=True)
            & pl.col("result_value").str.contains("/", literal=True)
        )
        .with_columns(pl.col("result_value").str.split("/").alias("bp_parts"))
        .filter(pl.col("bp_parts").list.len() == 2)
        .with_columns(
            pl.col("bp_parts").list.get(0).cast(pl.Float32, strict=False).alias("SYSTOLIC_BLOOD_PRESSURE_MMHG"),
            pl.col("bp_parts").list.get(1).cast(pl.Float32, strict=False).alias("DIASTOLIC_BLOOD_PRESSURE_MMHG"),
        )
        .unpivot(
            index=["subject_id", "chartdate", "seq_num"],
            on=["SYSTOLIC_BLOOD_PRESSURE_MMHG", "DIASTOLIC_BLOOD_PRESSURE_MMHG"],
            variable_name="result_name",
            value_name="result_num",
        )
        .filter(pl.col("result_num").is_not_null())
        .select(output_cols)
    )

    # Non-BP rows: cast to float and strip parentheses from result_name so
    # downstream configs can match on e.g. "BMI kg/m2" rather than "BMI (kg/m2)"
    non_bp_df = (
        omr_df.filter(pl.col("result_name").is_in(["BMI (kg/m2)", "Height (Inches)", "Weight (Lbs)"]))
        .with_columns(
            pl.col("result_value").cast(pl.Float32, strict=False).alias("result_num"),
            pl.col("result_name").str.replace_all(r"[()]", ""),
        )
        .filter(pl.col("result_num").is_not_null())
        .select(output_cols)
    )

    combined = pl.concat([bp_df, non_bp_df])
    return combined.pipe(_convert_date_to_end_of_day_timestamp)


def process_patients_df(patients_df: pl.LazyFrame, death_times_df: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregates death times from `admissions` and uses `dod` as fallback setting the time to 23:59:59,
    calculates birth year from `anchor_age` and `anchor_year`.
    """
    # Parse datetime once before aggregation for better performance
    # Cast handles both string and pre-parsed Datetime inputs
    death_times_df = (
        death_times_df.with_columns(pl.col("deathtime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
        .group_by("subject_id")
        .agg(pl.col("deathtime").min())
    )

    return patients_df.join(death_times_df, on="subject_id", how="left").select(
        "subject_id",
        # If we have a deathtime from admissions, use it; otherwise use dod
        pl.coalesce(
            pl.col("deathtime"),
            # dod is date only — cast to string first to handle pre-parsed Date inputs,
            # then parse and combine with end-of-day time
            pl.col("dod").cast(pl.String).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).dt.combine(END_OF_DAY),
        ).alias("dod"),
        (pl.col("anchor_year") - pl.col("anchor_age")).cast(pl.String).alias("year_of_birth"),
    )


def process_icu_chart_events_df(icu_chart_events_df: pl.LazyFrame, d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins ICU chart events with `d_items` to add item labels, sets null valueuom to empty string,
    and sets `value` to empty string when `valuenum` is not null.
    """
    return (
        _join_with_items_lookup(icu_chart_events_df, d_items_df, ["label"])
        .with_columns(pl.col("valueuom").fill_null(""))
        .pipe(_set_value_empty_string_when_valuenum_present)
    )


def process_icu_procedure_events_df(icu_procedure_events_df: pl.LazyFrame, d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins ICU procedure events with `d_items` to add item labels."""
    return _join_with_items_lookup(icu_procedure_events_df, d_items_df, ["label"])


def process_icu_input_events_df(icu_input_events_df: pl.LazyFrame, d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins ICU input events with `d_items` to add item labels and sets null rateuom and amountuom to empty string."""
    return _join_with_items_lookup(icu_input_events_df, d_items_df, ["label"]).with_columns(
        pl.col("rateuom").fill_null(""),
        pl.col("amountuom").fill_null(""),
    )


def process_icu_output_events_df(icu_output_events_df: pl.LazyFrame, d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Joins ICU output events with `d_items` to add item labels and fills null `valueuom` with empty strings."""
    return _join_with_items_lookup(icu_output_events_df, d_items_df, ["label"]).with_columns(
        pl.col("valueuom").fill_null("")
    )


# Processing functions registry for MIMIC-IV data files.
# Format: {
#     "relative/path/to/file": (
#         processing_function,
#         [("dependency/path", ["required", "columns"]), ...] or None
#     )
# }
# - Key: Relative path from MIMIC data root (e.g., "hosp/admissions")
# - Value: Tuple of (function, dependencies)
#   - function: Callable that processes the dataframe
#   - dependencies: List of (path, columns) tuples for required reference tables,
#                   or None if no dependencies needed
# NOTE: Dependencies are passed to the function as positional arguments in
# list order. Reordering entries will change which dataframe each parameter receives.
FUNCTIONS: Mapping[str, tuple[Callable[..., pl.LazyFrame], list[tuple[str, list[str]]] | None]] = {
    "hosp/admissions": (
        process_admissions_df,
        [("hosp/patients", ["subject_id", "gender"])],
    ),
    "hosp/diagnoses_icd": (
        process_icd_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_diagnoses", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/procedures_icd": (
        process_icd_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_procedures", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/drgcodes": (
        process_drgcodes_df,
        [("hosp/admissions", ["hadm_id", "dischtime"])],
    ),
    "hosp/hcpcsevents": (
        process_hcpcs_events_df,
        [("hosp/d_hcpcs", ["code", "long_description", "short_description"])],
    ),
    "hosp/labevents": (
        process_lab_events_df,
        [("hosp/d_labitems", ["itemid", "label", "fluid"])],
    ),
    "hosp/omr": (process_omr_df, None),
    "hosp/patients": (
        process_patients_df,
        [("hosp/admissions", ["subject_id", "deathtime"])],
    ),
    "icu/chartevents": (
        process_icu_chart_events_df,
        [("icu/d_items", ["itemid", "label"])],
    ),
    "icu/procedureevents": (
        process_icu_procedure_events_df,
        [("icu/d_items", ["itemid", "label"])],
    ),
    "icu/inputevents": (
        process_icu_input_events_df,
        [("icu/d_items", ["itemid", "label"])],
    ),
    "icu/outputevents": (
        process_icu_output_events_df,
        [("icu/d_items", ["itemid", "label"])],
    ),
}


def _discover_files(input_dir: Path) -> dict[str, tuple[Path, Callable[..., pl.LazyFrame]]]:
    """Discovers all unique data files in the input directory, returning a mapping of prefix to (path, read_fn)."""
    files: dict[str, tuple[Path, Callable[..., pl.LazyFrame]]] = {}
    for in_fp in input_dir.rglob("*"):
        if not in_fp.is_file() or in_fp.name.startswith("."):
            continue
        pfx = get_shard_prefix(input_dir, in_fp)
        if pfx in files:
            continue
        try:
            fp, read_fn = get_supported_fp(input_dir, pfx)
        except FileNotFoundError:
            logger.info(f"Skipping {pfx} @ {in_fp.resolve()} as no compatible dataframe file was found.")
            continue
        # For CSV files, increase schema inference to avoid misdetected types (e.g. float inferred as i64)
        if fp.name.endswith((".csv", ".csv.gz")):
            read_fn = partial(read_fn, infer_schema_length=10_000)
        files[pfx] = (fp, read_fn)
    return files


def _symlink_or_copy(src: Path, dst: Path, *, do_copy: bool) -> None:
    """Copies or symlinks a file, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        logger.info(f"No function needed: Copying {src.resolve()} to {dst.resolve()}")
        shutil.copy(src, dst)
    else:
        logger.info(f"No function needed: Symlinking {src.name} to output")
        relative_src = src.resolve().relative_to(dst.resolve().parent, walk_up=True)
        dst.symlink_to(relative_src)


def _load_dependencies(input_dir: Path) -> dict[str, pl.LazyFrame]:
    """Loads all dependency dataframes required by entries in FUNCTIONS."""
    # Collect the union of columns needed for each dependency prefix
    needed: dict[str, set[str]] = {}
    for _, deps in FUNCTIONS.values():
        if not deps:
            continue
        for dep_pfx, dep_cols in deps:
            needed.setdefault(dep_pfx, set()).update(dep_cols)

    loaded: dict[str, pl.LazyFrame] = {}
    for dep_pfx, cols in needed.items():
        try:
            fp, read_fn = get_supported_fp(input_dir, dep_pfx)
        except FileNotFoundError:
            logger.error(f"Dependency {dep_pfx} not found in {input_dir}. Required by FUNCTIONS entries.")
            raise
        st = datetime.now(tz=timezone.utc)
        logger.info(f"Loading dependency {dep_pfx}...")

        read_kwargs: dict[str, object] = {}
        # ICD tables need string schema to preserve leading zeros
        if dep_pfx in ("hosp/d_icd_diagnoses", "hosp/d_icd_procedures"):
            read_kwargs["schema_overrides"] = {"icd_code": pl.String, "icd_version": pl.String}
        # Load only needed columns for compressed files to save memory
        if fp.name.endswith(".csv.gz"):
            read_kwargs["columns"] = list(cols)

        loaded[dep_pfx] = cast(pl.LazyFrame, read_fn(fp, **read_kwargs)).select(list(cols))
        logger.info(f"  Loaded in {datetime.now(tz=timezone.utc) - st}")
    return loaded


def _process_file(
    pfx: str,
    fp: Path,
    read_fn: Callable[..., pl.LazyFrame],
    out_fp: Path,
    loaded_deps: dict[str, pl.LazyFrame],
) -> None:
    """Reads, processes, and writes a single file using its FUNCTIONS entry."""
    fn, deps = FUNCTIONS[pfx]
    st = datetime.now(tz=timezone.utc)
    logger.info(f"Processing {pfx}...")
    df = read_fn(fp)
    logger.info(f"  Loaded in {datetime.now(tz=timezone.utc) - st}")

    if deps:
        dep_dfs = [loaded_deps[dep_pfx] for dep_pfx, _ in deps]
        processed_df = fn(df, *dep_dfs)
    else:
        processed_df = fn(df)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    write_lazyframe(processed_df, out_fp)
    logger.info(f"  Wrote to {out_fp.name} in {datetime.now(tz=timezone.utc) - st}")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pre_MEDS",
)
def main(cfg: DictConfig) -> None:
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Discovers all data files in ``input_dir``, applies any registered
    processing function from ``FUNCTIONS``, and writes results to
    ``output_dir``.  Files without a registered function are symlinked
    (or copied when ``do_copy`` is set).
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    done_fp = output_dir / ".done"
    if done_fp.is_file() and not cfg.do_overwrite:
        logger.info(
            f"Pre-MEDS transformation already complete as {done_fp} exists and "
            f"do_overwrite={cfg.do_overwrite}. Returning."
        )
        return

    files = _discover_files(input_dir)
    loaded_deps = _load_dependencies(input_dir)

    for pfx, (fp, read_fn) in files.items():
        out_fp = output_dir / (f"{pfx}.parquet" if pfx in FUNCTIONS else fp.relative_to(input_dir))

        if out_fp.is_file():
            logger.debug(f"Already processed {pfx}, skipping")
            continue

        if pfx not in FUNCTIONS:
            _symlink_or_copy(fp, out_fp, do_copy=bool(cfg.do_copy))
        else:
            _process_file(pfx, fp, read_fn, out_fp, loaded_deps)

    done_fp.write_text(f"Finished at {datetime.now(tz=timezone.utc)}")
    logger.info(f"Pre-MEDS processing complete. Output in {output_dir}")


if __name__ == "__main__":
    main()
