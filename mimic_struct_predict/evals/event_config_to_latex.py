import argparse
import logging
import os
import re
from pathlib import Path

import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_event_configs(yaml_path: Path) -> pl.DataFrame:
    """Reads event_configs.yaml and returns a DataFrame of event configurations."""
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    rows = []
    for table, events in config.items():
        if not isinstance(events, dict):
            continue
        for event_name, event_cfg in events.items():
            if not isinstance(event_cfg, dict):
                continue
            time_val = event_cfg.get("time", "")
            raw_code = event_cfg.get("code", [])
            code_str = r" \newline ".join(str(c) for c in raw_code) if isinstance(raw_code, list) else str(raw_code)
            num_val = event_cfg.get("numeric_value")

            rows.append(
                {
                    "table": str(table),
                    "time": str(time_val),
                    "code": code_str,
                    "numeric value": str(num_val) if num_val is not None else "",
                }
            )
    return pl.DataFrame(rows)


def format_latex_cell(val: str, col: str) -> str:
    s = str(val).replace("_", r"\_")
    if col == "table":
        return r"\texttt{" + s + "}"
    return re.sub(r"col\((.*?)\)", lambda m: r"\texttt{col(" + m.group(1) + ")}", s)


def save_latex_table(df: pl.DataFrame, out_path: Path) -> None:
    """Generates a camera-ready LaTeX table using pandas DataFrame.to_latex()."""
    pdf = df.to_pandas()
    for col in pdf.columns:
        pdf[col] = pdf[col].apply(lambda s: format_latex_cell(s, col))

    latex_table = pdf.to_latex(index=False, escape=False)
    out_path.write_text(latex_table)
    log.info(f"Saved LaTeX table to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MEDS event configs to LaTeX table.")
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("mimic_struct_predict/preprocessing/meds/config/event_configs.yaml"),
        help="Path to event_configs.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "meds_config",
        help="Output directory (defaults to eval_dir / 'meds_config')",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(args)

    df = load_event_configs(args.config_path)

    save_latex_table(df, args.output_dir / "event_configs.tex")
    log.info(f"Saved summary tables to {args.output_dir}")


if __name__ == "__main__":
    main()
