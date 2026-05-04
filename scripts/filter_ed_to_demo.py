"""Filter MIMIC-IV ED tables to the demo cohort.

Overwrites every ``*.csv.gz`` file in ``<mimic_raw_dir>/ed`` by keeping only rows
with ``subject_id`` in ``data/raw_input/demo_subject_id.csv``.

Run:
    uv run python scripts/filter_ed_to_demo.py mimic_raw_dir=/path/to/MIMIC_RAW_DIR
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


def main(mimic_raw_dir: Path) -> None:
    ed_dir = mimic_raw_dir / "ed"
    if not ed_dir.is_dir():
        raise FileNotFoundError(f"ED directory not found: {ed_dir}")

    demo_subject_id_csv = mimic_raw_dir / "demo_subject_id.csv"
    if not demo_subject_id_csv.is_file():
        raise FileNotFoundError(f"Demo subject ID file not found: {demo_subject_id_csv}")

    # Read demo IDs and ensure they are nullable integers to match MIMIC schema
    demo_ids = pd.read_csv(demo_subject_id_csv)["subject_id"].astype("Int64")

    fps = sorted(ed_dir.glob("*.csv.gz"))

    for fp in fps:
        print(f"Processing {fp}...")
        tmp_fp = fp.with_name(fp.name + ".tmp")
        if tmp_fp.exists():
            tmp_fp.unlink()

        # Pandas handles gzip transparently out of the box
        # low_memory=False prevents warnings if Pandas struggles to guess column types
        df = pd.read_csv(str(fp), compression="infer", low_memory=False)
        original_count = len(df)

        # Filter down to the demo cohort
        df = df[df["subject_id"].isin(demo_ids)]

        # Write straight to disk with transparent compression
        df.to_csv(tmp_fp, index=False, compression="gzip")

        os.replace(tmp_fp, fp)
        print(f"Filtered {fp} to demo data (kept {len(df):,} out of {original_count:,} rows).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/filter_ed_to_demo.py mimic_raw_dir=/path/to/MIMIC_RAW_DIR")
        sys.exit(1)

    # Handle both `mimic_raw_dir=/path` and `/path` syntaxes gracefully
    arg = sys.argv[1]
    if arg.startswith("mimic_raw_dir="):
        arg = arg.split("=", 1)[1]

    mimic_raw_dir = Path(arg)

    main(mimic_raw_dir)
