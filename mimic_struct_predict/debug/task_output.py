import argparse
from pathlib import Path

import datasets
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--debug_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--structured", action="store_true", help="Convert ids to tokens and join with //")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    debug_dir = Path(args.debug_dir)

    if not task_dir.exists():
        print(f"Output directory does not exist: {task_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Iterate over all task directories
    task_dirs = [d for d in task_dir.iterdir() if d.is_dir()]

    for t_dir in task_dirs:
        task_name = t_dir.name
        train_dir = t_dir / "train"

        if not train_dir.exists() or not (train_dir / "dataset_info.json").exists():
            continue

        dataset = datasets.Dataset.load_from_disk(str(train_dir))

        if "input_ids" not in dataset.column_names:
            print(f"Column 'input_ids' not found in task {task_name}, skipping.")
            continue

        task_debug_dir = debug_dir / task_name
        task_debug_dir.mkdir(parents=True, exist_ok=True)

        subject_ids = sorted(set(dataset["subject_id"]))[:10]

        for subject_id in subject_ids:
            subject_dataset = dataset.filter(lambda example: example["subject_id"] == subject_id)
            subject_dataset = subject_dataset.sort("prediction_time")
            input_ids = subject_dataset["input_ids"][0]

            output_file = task_debug_dir / f"{subject_id}.md"
            if args.structured:
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                decoded_text = "//" + "//".join(tokens)
            else:
                decoded_text = tokenizer.decode(input_ids)
            output_file.write_text(decoded_text, encoding="utf-8")


if __name__ == "__main__":
    main()
