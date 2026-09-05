#!/usr/bin/env python
# Modified from https://github.com/ipolharvard/ethos-ares/blob/2d54383997318eb52f3d47b5969a66fc166b71ff/scripts/meds/mimic/pre_MEDS.py
#
# Each preprocessing function outlines the steps it takes in its docstring


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

ED_MEASUREMENT_UNIT_MAP = {
    "temperature": "Celsius",
    "heartrate": "bpm",
    "resprate": "bpm",
    "o2sat": "%",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "pain": "0 Low - 10 High",
    "acuity": "1 Highest - 5 Lowest Priority",
}

ED_MEASUREMENT_NAME_MAP = {
    "temperature": "Temperature",
    "heartrate": "Heart Rate",
    "resprate": "Respiratory Rate",
    "o2sat": "O2 Saturation",
    "sbp": "Systolic Blood Pressure",
    "dbp": "Diastolic Blood Pressure",
    "pain": "Pain Score",
    "acuity": "Emergency Severity Index",
}

GENDER_MAP = {"F": "Female", "M": "Male"}

PLACEHOLDER = "Placeholder"

# Map from https://mimic.mit.edu/docs/iv/modules/hosp/services.html
SERVICE_MAP = {
    "CMED": "Cardiac Medical",
    "CSURG": "Cardiac Surgery",
    "DENT": "Dental",
    "ENT": "Ear, Nose, and Throat",
    "EYE": "Eye Diseases",
    "GU": "Genitourinary",
    "GYN": "Gynecological",
    "MED": "Medical",
    "NB": "Newborn",
    "NBB": "Newborn Baby",
    "NMED": "Neurologic Medical",
    "NSURG": "Neurologic Surgical",
    "OBS": "Obstetrics",
    "ORTHO": "Orthopaedic",
    "OMED": "Oncologic Medical",
    "PSURG": "Plastic",
    "PSYCH": "Psychiatric",
    "SURG": "Surgical",
    "TRAUM": "Trauma",
    "TSURG": "Thoracic Surgical",
    "VSURG": "Vascular Surgical",
}

POUNDS_TO_KG = 0.45359237

NA_STRING = "NA"

### Helper Functions ###


def _unpivot_measurements(
    df: pl.LazyFrame,
    measurement_cols: list[str],
    index_cols: list[str],
) -> pl.LazyFrame:
    """Unpivots measurement columns into name, unit, value, and value_num."""
    schema = df.collect_schema()
    schema_names = schema.names()

    # Separate numeric from string columns to avoid schema conflicts during unpivot
    numeric_cols = [c for c in measurement_cols if c in schema_names and schema[c].is_numeric()]
    string_cols = [c for c in measurement_cols if c in schema_names and not schema[c].is_numeric()]

    dfs = []

    if numeric_cols:
        num_df = (
            df.unpivot(index=index_cols, on=numeric_cols, variable_name="name", value_name="value_num")
            .with_columns(
                # Set value to PLACEHOLDER when value_num exists MEDS pipeline will then remove this allowing value_num
                # to be appended -> Name//Unit//value_num
                # Otherwise None and so will be replaced with Unknown by MEDS -> Name//Unit//Unknown
                pl.when(pl.col("value_num").is_not_null())
                .then(pl.lit(PLACEHOLDER))
                .otherwise(pl.lit(None).cast(pl.String))
                .alias("value"),
                pl.col("value_num").cast(pl.Float32),
            )
            .select([*index_cols, "name", "value", "value_num"])
        )
        dfs.append(num_df)

    if string_cols:
        str_df = (
            df.unpivot(index=index_cols, on=string_cols, variable_name="name", value_name="value")
            .with_columns(pl.lit(None).cast(pl.Float32).alias("value_num"), pl.col("value").cast(pl.String))
            .select([*index_cols, "name", "value", "value_num"])
        )
        dfs.append(str_df)

    combined = pl.concat(dfs, how="vertical")

    return combined.with_columns(
        pl.col("name").replace_strict(ED_MEASUREMENT_UNIT_MAP, default="").alias("unit"),
        pl.col("name").replace_strict(ED_MEASUREMENT_NAME_MAP, default="").alias("name"),
    ).select([*index_cols, "name", "unit", "value", "value_num"])


def _convert_date_to_end_of_day_timestamp(df: pl.LazyFrame, date_column: str = "chartdate") -> pl.LazyFrame:
    """Converts a date-only column to a timestamp set at the end of the day."""
    return df.with_columns(pl.col(date_column).str.to_date().dt.combine(END_OF_DAY).alias(date_column))


def _join_with_items_lookup(
    events_df: pl.LazyFrame, d_items_df: pl.LazyFrame, columns_to_add: list[str]
) -> pl.LazyFrame:
    """Joins events with an items lookup table (e.g. `d_labitems`, `d_items`) by `itemid`."""
    d_items_df = d_items_df.select("itemid", *columns_to_add)
    return events_df.join(d_items_df, on="itemid", how="left")


def _add_time_by_id(df: pl.LazyFrame, time_source_df: pl.LazyFrame, on: str, time_column_name: str) -> pl.LazyFrame:
    """Joins the two dataframes by ``on`` and adds the time to the original dataframe."""

    time_source_df = time_source_df.select(on, time_column_name)
    return df.join(time_source_df, on=on, how="left")


def _convert_fahrenheit_to_celsius(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert Fahrenheit temperatures to Celsius (assumes Fahrenheit if > 45)."""
    return df.with_columns(
        pl.when(pl.col("temperature") > 45)
        .then((pl.col("temperature") - 32) * 5 / 9)
        .otherwise(pl.col("temperature"))
        .alias("temperature")
    )


def _join_icd_code_version_to_title(icd_df: pl.LazyFrame, d_icd_df: pl.LazyFrame) -> pl.LazyFrame:
    """Adds icd long titles based on icd code and version."""
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
    return icd_df


### Process Functions ###


def process_ed_diagnosis_df(
    ed_diagnosis_df: pl.LazyFrame, ed_edstays_df: pl.LazyFrame, hosp_d_icd_diagnoses_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Process ED Diagnosis DataFrame:
    - Add edstays outtime.
    - Join on icd code and version to get long_title.
    """
    return ed_diagnosis_df.pipe(_add_time_by_id, ed_edstays_df, on="stay_id", time_column_name="outtime").pipe(
        _join_icd_code_version_to_title, hosp_d_icd_diagnoses_df
    )


def process_ed_edstays_df(ed_edstays_df: pl.LazyFrame, hosp_patients_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ED Stays DataFrame:
    - Join patient df to get "anchor_year", "anchor_age".
    - Map gender F -> Female, M -> Male.
    - race, arrival_transport and disposition to titlecase.
    - Calc age at admission as anchor_age + intime year - anchor_year.
    """
    return (
        ed_edstays_df.join(hosp_patients_df, on="subject_id", how="left")
        .with_columns(
            pl.col("gender").replace_strict(GENDER_MAP),
            pl.col("race", "arrival_transport", "disposition").str.to_titlecase(),
            age=pl.col("anchor_age") + (pl.col("intime").str.to_datetime().dt.year() - pl.col("anchor_year")),
        )
        .drop(["anchor_age", "anchor_year"])
    )


def process_ed_medrecon_df(ed_medrecon_df: pl.LazyFrame, ed_edstays_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ED Medication Reconciliation DataFrame:
    - Add intime from edstays.
    - name to titlecase
    """
    return ed_medrecon_df.pipe(_add_time_by_id, ed_edstays_df, on="stay_id", time_column_name="intime").with_columns(
        pl.col("name").str.to_titlecase()
    )


def process_ed_triage_df(ed_triage_df: pl.LazyFrame, ed_edstays_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ED Triage DataFrame:
    - Convert Fahrenheit temperatures to Celsius.
    - Add intime from edstays.
    - Unpivot measurements into rows and set unit based on mapping.
    - Split chiefcomplaint by "," into multiple rows.
    """
    ed_triage_df = ed_triage_df.pipe(_convert_fahrenheit_to_celsius).pipe(
        _add_time_by_id, ed_edstays_df, on="stay_id", time_column_name="intime"
    )

    vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
    df_vitals = _unpivot_measurements(ed_triage_df, vital_cols, ["subject_id", "intime"])

    # Efficiently split comma-separated values into multiple rows
    df_cc = (
        ed_triage_df.select(
            "subject_id",
            "intime",
            name=pl.lit("Chief Complaint"),
            unit=pl.lit("NA"),
            value=pl.col("chiefcomplaint").cast(pl.String).str.split(","),
            value_num=pl.lit(None).cast(pl.Float32),
        )
        .explode("value")
        .with_columns(pl.col("value").str.strip_chars().str.to_titlecase())
    )

    return pl.concat([df_vitals, df_cc], how="vertical")


def process_ed_vitalsign_df(ed_vitalsign_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ED Vitals DataFrame:
    - Convert Fahrenheit temperatures to Celsius.
    - Unpivot vital signs into rows.
    - Set unit based on mapping.
    - Create value and value_num columns based on data type.
    - Value to titlecase.

    Note: rythmn col not included as only in ~3% of rows as Sinus Tachycardia (high heartrate) which is known
    from the heartrate value.
    """
    vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
    return (
        ed_vitalsign_df.pipe(_convert_fahrenheit_to_celsius)
        .pipe(_unpivot_measurements, vital_cols, ["subject_id", "charttime"])
        .with_columns(pl.col("value").str.to_titlecase())
    )


def process_icu_chartevents_df(icu_chartevents_df: pl.LazyFrame, icu_d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ICU Chartevents DataFrame:
    - Join with d_items_df to add label.
    - Set "___" to null for 'value'.
    - Set valueuom to NA when null.
    - Values:
        - When valuenum is present we will use it as a numeric value and set value string to PLACEHOLDER.
        - When valuenum is null, use value as a string.
    """
    return _join_with_items_lookup(icu_chartevents_df, icu_d_items_df, ["label"]).with_columns(
        pl.when(pl.col("valuenum").is_not_null())
        .then(pl.lit(PLACEHOLDER))
        .otherwise(pl.col("value").replace("___", None))
        .alias("value"),
        pl.col("valueuom").fill_null(NA_STRING),
    )


def process_icu_procedureevents_df(icu_procedureevents_df: pl.LazyFrame, icu_d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ICU Procedureevents DataFrame:
    - Join with d_items_df to add label.
    """
    return _join_with_items_lookup(icu_procedureevents_df, icu_d_items_df, ["label"])


def process_icu_inputevents_df(icu_inputevents_df: pl.LazyFrame, icu_d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ICU Inputevents DataFrame:
    - Join with d_items_df to add label.
    """
    return _join_with_items_lookup(icu_inputevents_df, icu_d_items_df, ["label"])


def process_icu_outputevents_df(icu_outputevents_df: pl.LazyFrame, icu_d_items_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process ICU Outputevents DataFrame:
    - Join with d_items_df to add label.
    """
    return _join_with_items_lookup(icu_outputevents_df, icu_d_items_df, ["label"])


def process_hosp_admissions_df(hosp_admissions_df: pl.LazyFrame, hosp_patients_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital Admissions DataFrame:
    - Map patient_df gender: F → Female, M → Male.
    - To title case admission_type, admission_location, discharge_location, insurance, language, marital_status, race.
    - Join admission and patient df to get "gender", "anchor_year", "anchor_age".
    - Calc age at admission as anchor_age + admittime year - anchor_year.
    """
    patients_clean = hosp_patients_df.select(
        "subject_id",
        pl.col("gender").replace_strict(GENDER_MAP),
        "anchor_year",
        "anchor_age",
    )

    return (
        hosp_admissions_df.join(patients_clean, on="subject_id", how="left")
        .with_columns(
            pl.col(
                "admission_type",
                "admission_location",
                "discharge_location",
                "insurance",
                "language",
                "marital_status",
                "race",
            ).str.to_titlecase(),
            age=pl.col("anchor_age") + (pl.col("admittime").str.to_datetime().dt.year() - pl.col("anchor_year")),
        )
        .drop(["anchor_age", "anchor_year"])
    )


def process_hosp_diagnosis_df(
    hosp_diagnosis_df: pl.LazyFrame, hosp_admissions_df: pl.LazyFrame, hosp_d_icd_diagnoses_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Process Hospital Diagnosis DataFrame:
    - Add admission dischtime.
    - Join on icd code and version to get long_title.
    """
    return hosp_diagnosis_df.pipe(_add_time_by_id, hosp_admissions_df, on="hadm_id", time_column_name="dischtime").pipe(
        _join_icd_code_version_to_title, hosp_d_icd_diagnoses_df
    )


def process_hosp_drgcodes_df(hosp_drgcodes_df: pl.LazyFrame, hosp_admissions_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital DRGCodes (Diagnosis Related Groups) DataFrame:
    - Map drg_severity and drg_mortality to free text categorical values.
    - Description to titlecase.
    - Add admission dischtime.
    """
    # Map from https://hcup-us.ahrq.gov/db/nation/nis/APR-DRGsV20MethodologyOverviewandBibliography.pdf p.21
    severity_map = {"1": "Minor", "2": "Moderate", "3": "Major", "4": "Extreme"}
    return hosp_drgcodes_df.with_columns(
        pl.col("drg_severity", "drg_mortality").replace_strict(severity_map, default=pl.lit(NA_STRING)),
        pl.col("description").str.to_titlecase(),
    ).pipe(_add_time_by_id, hosp_admissions_df, on="hadm_id", time_column_name="dischtime")


def process_hosp_hcpcs_events_df(hosp_hcpcs_events_df: pl.LazyFrame, hosp_d_hcpcs_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital HCPCS (Healthcare Common Procedure Coding System) DataFrame:
    - Join with d_hcpcs to add procedure descriptions (preferring long, falling back to short).
    - Convert chartdate to end-of-day timestamps.
    """
    hosp_d_hcpcs_df = hosp_d_hcpcs_df.select(
        "code",
        pl.coalesce("long_description", "short_description").alias("long_description"),
    )
    return hosp_hcpcs_events_df.join(hosp_d_hcpcs_df, left_on="hcpcs_cd", right_on="code", how="left").pipe(
        _convert_date_to_end_of_day_timestamp
    )


def process_hosp_lab_events_df(hosp_lab_events_df: pl.LazyFrame, hosp_d_labitems_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital Lab Events DataFrame:
    - Join with d_labitems_df to add fluid and label.
    - Set null and " " units to NA_STRING.
    - Map priority stat -> urgent and to titlecase.
    - Value:
        - When valuenum is present we will use it as a numeric value and set value string to PLACEHOLDER.
        - When valuenum is null,
            - Use value if not null or "___"
            - Otherwise use comments (if not > 50 char)
            - Otherwise null
        - Set to title case
    """
    return _join_with_items_lookup(hosp_lab_events_df, hosp_d_labitems_df, ["fluid", "label"]).with_columns(
        pl.col("valueuom").fill_null(NA_STRING).replace(" ", NA_STRING),
        pl.col("priority").replace_strict({"STAT": "URGENT"}, default=pl.col("priority")).str.to_titlecase(),
        pl.when(pl.col("valuenum").is_not_null())
        .then(pl.lit(PLACEHOLDER))
        .otherwise(
            pl.coalesce(
                pl.col("value").replace("___", None),
                pl.when(pl.col("comments").str.len_chars() > 50)
                .then(None)
                .otherwise(pl.col("comments").replace("___", None)),
            )
        )
        .str.to_titlecase()
        .alias("value"),
    )


def process_hosp_emar_df(hosp_emar_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital EMAR (electronic medication administration record) DataFrame:
    - Drop any rows where medication column is null.
    - Fill any null event_txt with NA_STRING.
    """
    return hosp_emar_df.drop_nulls(subset=["medication"]).with_columns(pl.col("event_txt").fill_null(NA_STRING))


def process_hosp_omr_df(hosp_omr_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital OMR (Online Medical Record ~misc observations) DataFrame:
    - Blood pressure:
        - Split into systolic and diastolic rows.
        - Set unit to "mmHg".
    - BMI, weight, height:
        - Convert result_value to float.
        - Extract unit into result_unit and keep category in result_name.
    - Merge BP and BMI/weight/height.
    - Convert weight from lbs to kg.
    - Convert chartdate to EOD.
    """
    output_cols = ["subject_id", "chartdate", "seq_num", "result_name", "result_num", "result_unit"]

    # Blood pressure: split on "/" and unpivot into separate rows
    bp_df = (
        hosp_omr_df.filter(
            pl.col("result_name").str.contains("Blood Pressure", literal=True)
            & pl.col("result_value").str.contains("/", literal=True)
        )
        .with_columns(pl.col("result_value").str.split("/").alias("bp_parts"))
        .filter(pl.col("bp_parts").list.len() == 2)
        .with_columns(
            pl.col("bp_parts").list.get(0).cast(pl.Float32, strict=False).alias("Systolic Blood Pressure"),
            pl.col("bp_parts").list.get(1).cast(pl.Float32, strict=False).alias("Diastolic Blood Pressure"),
            result_unit=pl.lit("mmHg"),
        )
        .drop("result_name")
        .unpivot(
            index=["subject_id", "chartdate", "seq_num", "result_unit"],
            on=["Systolic Blood Pressure", "Diastolic Blood Pressure"],
            variable_name="result_name",
            value_name="result_num",
        )
        .select(output_cols)
    )

    # Non-BP rows: cast to float and extract unit from result_name
    non_bp_df = (
        hosp_omr_df.filter(pl.col("result_name").is_in(["BMI (kg/m2)", "Height (Inches)", "Weight (Lbs)"]))
        .with_columns(
            pl.col("result_value").cast(pl.Float32, strict=False).alias("result_num"),
            pl.col("result_name").str.extract(r" \((.*)\)").alias("result_unit"),
            pl.col("result_name").str.replace(r" \(.*\)", "").alias("result_name"),
        )
        .with_columns(
            pl.when(pl.col("result_name") == "Weight")
            .then(pl.col("result_num") * POUNDS_TO_KG)
            .otherwise(pl.col("result_num"))
            .alias("result_num"),
            pl.when(pl.col("result_name") == "Weight")
            .then(pl.lit("kg"))
            .otherwise(pl.col("result_unit"))
            .alias("result_unit"),
        )
        .select(output_cols)
    )

    return pl.concat([bp_df, non_bp_df]).pipe(_convert_date_to_end_of_day_timestamp)


def process_hosp_patients_df(hosp_patients_df: pl.LazyFrame, hosp_admissions_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital Patients DataFrame:
    - Take min death time from admissions_df if it exists.
    - If no death time in admissions, use dod (date only) as fallback and convert to end of day.
    """
    hosp_admissions_df = (
        hosp_admissions_df.with_columns(pl.col("deathtime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
        .group_by("subject_id")
        .agg(pl.col("deathtime").min())
    )

    return hosp_patients_df.join(hosp_admissions_df, on="subject_id", how="left").select(
        "subject_id",
        pl.coalesce(
            pl.col("deathtime"),
            pl.col("dod").cast(pl.String).str.to_date(format="%Y-%m-%d", strict=False).dt.combine(END_OF_DAY),
        ).alias("dod"),
    )


def process_hosp_pharmacy_df(hosp_pharmacy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital Pharmacy DataFrame:
    - Drop rows where medication is null.
    - Set frequency to titlecase.
    """
    return hosp_pharmacy_df.drop_nulls(subset=["medication"]).with_columns(pl.col("frequency").str.to_titlecase())


def process_hosp_procedures_df(
    hosp_procedures_df: pl.LazyFrame, hosp_admissions_df: pl.LazyFrame, hosp_d_icd_procedures_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Process Hospital Procedures DataFrame:
    - Add admission dischtime.
    - Join on icd code and version to get long_title.
    - Convert chartdate to end-of-day timestamps.
    """
    return (
        hosp_procedures_df.pipe(_add_time_by_id, hosp_admissions_df, on="hadm_id", time_column_name="dischtime")
        .pipe(_join_icd_code_version_to_title, hosp_d_icd_procedures_df)
        .pipe(_convert_date_to_end_of_day_timestamp)
    )


def process_hosp_transfers_df(hosp_transfers_df: pl.LazyFrame) -> pl.LazyFrame:
    """Process Hospital Transfers DataFrame:
    - Map eventtype to titlecase.
    """
    return hosp_transfers_df.with_columns(pl.col("eventtype").str.to_titlecase())


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
    "ed/diagnosis": (
        process_ed_diagnosis_df,
        [("ed/edstays", ["stay_id", "outtime"]), ("hosp/d_icd_diagnoses", ["icd_code", "icd_version", "long_title"])],
    ),
    "ed/edstays": (
        process_ed_edstays_df,
        [("hosp/patients", ["subject_id", "anchor_year", "anchor_age"])],
    ),
    "ed/medrecon": (
        process_ed_medrecon_df,
        [("ed/edstays", ["stay_id", "intime"])],
    ),
    "ed/triage": (
        process_ed_triage_df,
        [
            ("ed/edstays", ["stay_id", "intime"]),
        ],
    ),
    "ed/vitalsign": (
        process_ed_vitalsign_df,
        None,
    ),
    ### Hospital Tables ###
    "hosp/admissions": (
        process_hosp_admissions_df,
        [("hosp/patients", ["subject_id", "gender", "anchor_year", "anchor_age"])],
    ),
    "hosp/diagnoses_icd": (
        process_hosp_diagnosis_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_diagnoses", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/drgcodes": (
        process_hosp_drgcodes_df,
        [("hosp/admissions", ["hadm_id", "dischtime"])],
    ),
    "hosp/hcpcsevents": (
        process_hosp_hcpcs_events_df,
        [("hosp/d_hcpcs", ["code", "long_description", "short_description"])],
    ),
    "hosp/labevents": (
        process_hosp_lab_events_df,
        [("hosp/d_labitems", ["itemid", "label", "fluid"])],
    ),
    "hosp/emar": (process_hosp_emar_df, None),
    "hosp/omr": (process_hosp_omr_df, None),
    "hosp/patients": (
        process_hosp_patients_df,
        [("hosp/admissions", ["subject_id", "deathtime"])],
    ),
    "hosp/pharmacy": (process_hosp_pharmacy_df, None),
    "hosp/procedures_icd": (
        process_hosp_procedures_df,
        [
            ("hosp/admissions", ["hadm_id", "dischtime"]),
            ("hosp/d_icd_procedures", ["icd_code", "icd_version", "long_title"]),
        ],
    ),
    "hosp/transfers": (process_hosp_transfers_df, None),
    # ### ICU Tables ###
    # "icu/chartevents": (
    #     process_icu_chartevents_df,
    #     [("icu/d_items", ["itemid", "label"])],
    # ),
    # "icu/procedureevents": (
    #     process_icu_procedureevents_df,
    #     [("icu/d_items", ["itemid", "label"])],
    # ),
    # "icu/inputevents": (
    #     process_icu_inputevents_df,
    #     [("icu/d_items", ["itemid", "label"])],
    # ),
    # "icu/outputevents": (
    #     process_icu_outputevents_df,
    #     [("icu/d_items", ["itemid", "label"])],
    # ),
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
            read_kwargs: dict[str, object] = {"infer_schema_length": 100_000}
            # HACK: Pharamacy still has issues even at 100k rows so hardcode issue columns as strings
            # (we don't use these columns so format does not matter)
            if pfx == "hosp/pharmacy":
                read_kwargs["schema_overrides"] = {
                    "lockout_interval": pl.String,
                    "fill_quantity": pl.String,
                    "one_hr_max": pl.String,
                }
            read_fn = partial(read_fn, **read_kwargs)
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
