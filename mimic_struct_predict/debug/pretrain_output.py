import argparse
import logging
from pathlib import Path

import datasets
import polars as pl
from transformers import AutoTokenizer

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for debug outputs")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw pretraining parquet dataset")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Path to saved tokenizer or HF model name")
    parser.add_argument("--tokenized_dataset_dir", type=str, default=None, help="Path to huggingface tokenized dataset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_out_dir = output_dir / "raw"
    tok_out_dir = output_dir / "tokenized"

    raw_out_dir.mkdir(parents=True, exist_ok=True)
    if args.tokenizer_dir:
        tok_out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading raw data from {args.raw_data_dir}")
    df = pl.read_parquet(args.raw_data_dir)
    subjects = df["subject_id"].unique().sort().head(10)

    for subject_id in subjects.to_list():
        subject_df = df.filter(pl.col("subject_id") == subject_id)
        text = subject_df["ehr_text"][0]

        # Write raw output
        raw_file = raw_out_dir / f"{subject_id}.md"
        raw_file.write_text(text)

    log.info(f"Loading tokenizer from {args.tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    log.info(f"Loading tokenized dataset from {args.tokenized_dataset_dir}")
    dataset = datasets.load_from_disk(args.tokenized_dataset_dir)

    for i, subject_id in enumerate(subjects.to_list()):
        # Get the input_ids from the pre-computed huggingface dataset
        input_ids = dataset["train"][i]["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        tok_file = tok_out_dir / f"{subject_id}.md"
        tok_file.write_text(str(tokens))

    log.info(f"Saved {len(subjects)} debug outputs to {output_dir}")


if __name__ == "__main__":
    main()
