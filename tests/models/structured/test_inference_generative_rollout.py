from datetime import datetime, timedelta
from types import SimpleNamespace

from mimic_struct_predict.models.structured.inference_generative_rollout import (
    extract_last_category_content,
    parse_completion,
    parse_delta_line,
    parse_time_line,
)

LINE_DELIMITER = "\n"


def _build_text(lines: list[str]) -> str:
    return LINE_DELIMITER.join(lines)


def test_parse_time_line_parses_hour_and_minute() -> None:
    assert parse_time_line("### Time//09h//30m", "### Time//", "Unspecified", "//") == (9, 30)


def test_parse_time_line_returns_none_for_out_of_range_values() -> None:
    assert parse_time_line("### Time//25h//00m", "### Time//", "Unspecified", "//") is None
    assert parse_time_line("### Time//20h//60m", "### Time//", "Unspecified", "//") is None


def test_parse_delta_line_extracts_positive_day_count() -> None:
    assert parse_delta_line("## Delta//+3 days", "## Delta//") == 3


def test_get_last_header_line_returns_latest_category() -> None:
    lines = ["#### Labs", "A", "#### Medications", "B"]
    # use extract_last_category_content on joined text
    text = _build_text(lines) + LINE_DELIMITER
    assert extract_last_category_content(text, "#### ", LINE_DELIMITER) == "Medications"


def test_parse_output_text_parses_headers_times_and_deltas() -> None:
    input_time = datetime(2024, 1, 1, 8, 0)
    text = _build_text(
        [
            "#### Vitals",
            "### Time//09h//30m",
            "HeartRate",
            "## Delta//+2 days",
            "Temp",
        ]
    )

    tokenization_cfg = SimpleNamespace(
        line_delimiter=LINE_DELIMITER,
        delimiter="//",
        time_unspecified="Unspecified",
        prefixes=SimpleNamespace(category="#### ", time="### Time//", delta="## Delta//"),
    )

    timestamps, codes, found, finished = parse_completion(
        text,
        prediction_time=input_time,
        prediction_header="Initial",
        end_time_delta=timedelta(days=3),
        end_codes=[],
        tokenization_cfg=tokenization_cfg,
    )

    events = list(zip(timestamps, codes))

    assert events == [
        (datetime(2024, 1, 1, 9, 30), "Vitals//HeartRate"),
        (datetime(2024, 1, 3, 9, 30), "Vitals//Temp"),
    ]
    assert found is False
    assert finished is False


def test_parse_output_text_stops_when_target_code_is_found() -> None:
    input_time = datetime(2024, 1, 1, 8, 0)
    text = _build_text(["alpha", "beta", "gamma"])

    tokenization_cfg = SimpleNamespace(
        line_delimiter=LINE_DELIMITER,
        delimiter="//",
        time_unspecified="Unspecified",
        prefixes=SimpleNamespace(category="#### ", time="### Time//", delta="## Delta//"),
    )

    timestamps, codes, found, finished = parse_completion(
        text,
        prediction_time=input_time,
        prediction_header="Header",
        end_time_delta=timedelta(days=1),
        end_codes=["Header//beta"],
        tokenization_cfg=tokenization_cfg,
    )

    events = list(zip(timestamps, codes))

    assert events == [
        (input_time, "Header//alpha"),
        (input_time, "Header//beta"),
    ]
    assert found is True
    assert finished is True


def test_parse_output_text_marks_finished_when_time_exceeds_window() -> None:
    input_time = datetime(2024, 1, 1, 8, 0)
    text = _build_text(
        [
            "### Time//10h//00m",
            "event-after-window",
        ]
    )

    tokenization_cfg = SimpleNamespace(
        line_delimiter=LINE_DELIMITER,
        delimiter="//",
        time_unspecified="Unspecified",
        prefixes=SimpleNamespace(category="#### ", time="### Time//", delta="## Delta//"),
    )

    timestamps, codes, found, finished = parse_completion(
        text,
        prediction_time=input_time,
        prediction_header="Header",
        end_time_delta=timedelta(hours=1),
        end_codes=[],
        tokenization_cfg=tokenization_cfg,
    )

    events = list(zip(timestamps, codes))

    assert events == []
    assert found is False
    assert finished is True


def test_parse_output_text_ignores_invalid_time_and_keeps_current_time() -> None:
    input_time = datetime(2024, 1, 1, 8, 0)
    text = _build_text(
        [
            "### Time//not-a-time",
            "first",
            "### Time//25h//00m",
            "second",
        ]
    )

    tokenization_cfg = SimpleNamespace(
        line_delimiter=LINE_DELIMITER,
        delimiter="//",
        time_unspecified="Unspecified",
        prefixes=SimpleNamespace(category="#### ", time="### Time//", delta="## Delta//"),
    )

    timestamps, codes, found, finished = parse_completion(
        text,
        prediction_time=input_time,
        prediction_header="Header",
        end_time_delta=timedelta(days=1),
        end_codes=[],
        tokenization_cfg=tokenization_cfg,
    )

    events = list(zip(timestamps, codes))

    assert events == [
        (input_time, "Header//first"),
        (input_time, "Header//second"),
    ]
    assert found is False
    assert finished is False
