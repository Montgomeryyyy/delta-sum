"""Plot windows per patient and tokens per window chunk for natural-language EHR text.

Example:
    python -m mimic_struct_predict.visualize.visualize_chunks \
        --data_dir ${NATURAL_LANGUAGE_RAW_DATA_DIR} \
        --tokenizer /path/to/medgemma-27b-it \
        --output chunks_per_patient.png

The tokenizer is loaded from local files only (e.g. a downloaded model directory), so this runs offline.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SPLITS = ["train", "tuning"]
WINDOW_REGEX = r"^## New Month"
BATCH_SIZE = 1000

WINDOW_BINS = [
    (0, 0, "0"),
    (1, 2, "1–2"),
    (3, 4, "3–4"),
    (5, 6, "5–6"),
    (7, 10, "7–10"),
    (11, 20, "11–20"),
    (21, 50, "21–50"),
    (51, 100, "51–100"),
    (101, 199, "101–199"),
    (200, None, "200+"),
]

TOKEN_BREAKS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def split_into_windows(text: str, window_re: re.Pattern) -> list[str]:
    """Split text at each window marker; text before the first marker is kept with the first window."""
    starts = [match.start() for match in window_re.finditer(text)]
    if not starts:
        return []
    starts[0] = 0
    return [text[start:end] for start, end in zip(starts, starts[1:] + [len(text)])]


def count_tokens(texts: list[str], tokenizer) -> list[int]:
    n_tokens = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Tokenizing"):
        encoded = tokenizer(texts[i : i + BATCH_SIZE], add_special_tokens=False)["input_ids"]
        n_tokens.extend(len(ids) for ids in encoded)
    return n_tokens


def bin_label(n: int) -> str:
    for low, high, label in WINDOW_BINS:
        if n >= low and (high is None or n <= high):
            return label
    raise ValueError(f"No bin for {n}")


def plot(per_patient: pl.DataFrame, chunks: pl.DataFrame, output: Path) -> None:
    window_labels = [label for _, _, label in WINDOW_BINS]
    window_dist = per_patient.with_columns(
        pl.col("n_windows").map_elements(bin_label, return_dtype=pl.String).alias("bin")
    )["bin"].value_counts()
    window_heights = [window_dist.filter(pl.col("bin") == label)["count"].sum() for label in window_labels]

    token_labels = [f"≤{b}" for b in TOKEN_BREAKS] + [f">{TOKEN_BREAKS[-1]}"]
    token_dist = chunks.select(pl.col("n_tokens").cut(TOKEN_BREAKS, labels=token_labels).alias("bin"))[
        "bin"
    ].value_counts()
    token_heights = [token_dist.filter(pl.col("bin") == label)["count"].sum() for label in token_labels]

    # Cumulative share for the "≤" bins, plain share for the overflow bin.
    pct = []
    running = 0
    for height in token_heights[:-1]:
        running += height
        pct.append(100 * running / chunks.height)
    pct.append(100 * token_heights[-1] / chunks.height)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    axes[0].bar(window_labels, window_heights)
    axes[0].set_xlabel(f"Windows per patient ({WINDOW_REGEX})")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title(f"Windows per patient (n={per_patient.height})")
    axes[0].tick_params(axis="x", labelsize=8)

    bars = axes[1].bar(token_labels, token_heights)
    axes[1].set_xlabel(f"Tokens per window chunk ({WINDOW_REGEX})")
    axes[1].set_ylabel("Number of chunks")
    axes[1].set_title(f"Chunk length (n={chunks.height})")
    axes[1].set_ylim(0, max(token_heights) * 1.15)
    axes[1].tick_params(axis="x", labelrotation=30, labelsize=8)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")
    for bar, share in zip(bars, pct):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{share:.1f}%", ha="center", va="bottom", fontsize=8
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved figure to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True, help="Raw text dir with <split>/*.parquet.")
    parser.add_argument("--tokenizer", required=True, help="Local model or tokenizer directory.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    window_re = re.compile(WINDOW_REGEX, flags=re.MULTILINE)

    df = pl.concat(
        pl.read_parquet(args.data_dir / split / "*.parquet", columns=["subject_id", "ehr_text"]) for split in SPLITS
    )
    print(f"Loaded {df.height:,} patients from {SPLITS}")

    chunks = (
        df.select(
            pl.col("ehr_text")
            .map_elements(lambda text: split_into_windows(text, window_re), return_dtype=pl.List(pl.String))
            .alias("chunk")
        )
        .explode("chunk")
        .drop_nulls("chunk")
    )
    chunks = chunks.with_columns(pl.Series("n_tokens", count_tokens(chunks["chunk"].to_list(), tokenizer)))

    # Patients with no windows have no chunks, so count windows from the full patient table.
    per_patient = df.group_by("subject_id").agg(
        pl.col("ehr_text").str.count_matches(f"(?m){WINDOW_REGEX}").sum().alias("n_windows")
    )

    print(
        chunks.select(
            pl.len().alias("chunks"),
            pl.col("n_tokens").min().alias("min"),
            pl.col("n_tokens").median().alias("median"),
            pl.col("n_tokens").quantile(0.95).alias("p95"),
            pl.col("n_tokens").quantile(0.99).alias("p99"),
            pl.col("n_tokens").max().alias("max"),
        )
    )
    plot(per_patient, chunks, args.output)


if __name__ == "__main__":
    main()
