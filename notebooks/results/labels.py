import os
from pathlib import Path

import pandas as pd

# 1. Define the base data directory (adjust path if your script runs elsewhere)
base_dir = Path(os.getenv("SCRATCH_DIR")) / "data" / "aces_labels"


def analyze_split_labels(data_path):
    # Define the splits we want to check
    splits = ["train", "tuning", "held_out"]

    results = []

    print("Scanning split folders and processing Parquet files...\n")

    # 1. Iterate through each main label task folder
    for label_dir in sorted(data_path.iterdir()):
        if not label_dir.is_dir():
            continue

        # Initialize the base row for this label task
        row_data = {"Label Task": label_dir.name}

        # 2. Process each split separately
        for split in splits:
            split_dir = label_dir / split
            total_rows = 0
            total_positive = 0

            # Find all parquet files specifically within this split folder
            if split_dir.exists() and split_dir.is_dir():
                parquet_files = list(split_dir.rglob("*.parquet"))

                for file_path in parquet_files:
                    try:
                        df = pd.read_parquet(file_path)
                        total_rows += len(df)

                        if "boolean_value" in df.columns:
                            total_positive += df["boolean_value"].sum()
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            # Calculate positive percentage if rows exist
            pos_percentage = round((total_positive / total_rows) * 100, 2) if total_rows > 0 else 0.0

            # Clean up the name for column mapping (e.g., held_out -> Held Out)
            split_name = split.replace("_", " ").title()

            # Store the data for this split
            row_data[f"{split_name} Rows"] = total_rows
            row_data[f"{split_name} %"] = pos_percentage

        results.append(row_data)

    if not results:
        print("No label directories or data found.")
        return

    # 3. Print out the results as a neatly formatted multi-column table
    # Custom string formatting widths to ensure it fits well in terminal output
    header_fmt = "{:<38} | {:>10} | {:>8} | {:>11} | {:>9} | {:>13} | {:>11}"
    row_fmt = "{:<38} | {:>10,} | {:>7.2f}% | {:>11,} | {:>8.2f}% | {:>13,} | {:>10.2f}%"
    separator = "-" * 109

    print(separator)
    print(
        header_fmt.format(
            "Label Task",
            "Train Rows",
            "Train %",
            "Tuning Rows",
            "Tuning %",
            "Held Out Rows",
            "Held Out %",
        )
    )
    print(separator)

    for r in results:
        print(
            row_fmt.format(
                r["Label Task"],
                r["Train Rows"],
                r["Train %"],
                r["Tuning Rows"],
                r["Tuning %"],
                r["Held Out Rows"],
                r["Held Out %"],
            )
        )

    print(separator)


if __name__ == "__main__":
    analyze_split_labels(base_dir)
