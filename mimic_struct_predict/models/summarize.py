import logging
import re
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)


def generate_summaries(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
    """Run vLLM generation over prompts and return decoded summary texts."""
    if not prompts:
        return []
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]


def truncate_chunk_text(text: str, tokenizer, max_text_tokens: int) -> tuple[str, bool, int]:
    """Keep the last ``max_text_tokens`` of the EHR chunk (not the full prompt).

    Truncating the full prompt would drop the leading instruction for long notes.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)
    if n_tokens <= max_text_tokens:
        return text, False, n_tokens
    return (
        tokenizer.decode(token_ids[-max_text_tokens:], skip_special_tokens=True),
        True,
        n_tokens,
    )


def summarize_batch(
    batch: dict,
    prompt_template: str,
    llm: LLM,
    sampling_params: SamplingParams,
    max_prompt_tokens: int,
) -> tuple[dict, int]:
    """Add a ``summary`` field to each nested chunk for a batch of patients.

    Returns the updated batch and the number of prompts that needed truncation.
    """
    tokenizer = llm.get_tokenizer()
    # Reserve space for the instruction wrapper around {text}.
    wrapper_tokens = len(
        tokenizer.encode(prompt_template.format(text=""), add_special_tokens=False)
    )
    max_text_tokens = max(1, max_prompt_tokens - wrapper_tokens)

    prompts: list[str] = []
    locations: list[tuple[int, int]] = []
    truncated_flags: list[bool] = []
    n_truncated = 0

    for p_i, chunks in enumerate(batch["chunks"]):
        for c_i, chunk in enumerate(chunks):
            text, was_truncated, n_tokens = truncate_chunk_text(
                chunk["text"], tokenizer, max_text_tokens
            )
            if was_truncated:
                n_truncated += 1
                log.debug(
                    f"Truncated text for subject_id={batch['subject_id'][p_i]} "
                    f"chunk_id={chunk.get('chunk_id', c_i)}: {n_tokens} -> {max_text_tokens} tokens"
                )
            prompts.append(prompt_template.format(text=text))
            locations.append((p_i, c_i))
            truncated_flags.append(was_truncated)

    summaries = generate_summaries(llm, prompts, sampling_params)

    out_chunks: list[list[dict]] = []
    for chunks in batch["chunks"]:
        out_chunks.append([dict(chunk) for chunk in chunks])

    for (p_i, c_i), summary, was_truncated in zip(locations, summaries, truncated_flags, strict=True):
        out_chunks[p_i][c_i]["summary"] = summary
        out_chunks[p_i][c_i]["truncated"] = was_truncated

    return {"subject_id": batch["subject_id"], "chunks": out_chunks}, n_truncated


def summarize_split(
    split_dataset: datasets.Dataset,
    prompt_template: str,
    llm: LLM,
    sampling_params: SamplingParams,
    chunk_size: int,
    max_prompt_tokens: int,
) -> datasets.Dataset:
    """Summarize one split by patient batches."""
    subject_ids: list = []
    chunks_col: list[list[dict]] = []
    total = (len(split_dataset) + chunk_size - 1) // chunk_size
    n_truncated_total = 0
    n_prompts_total = 0

    for i in tqdm(range(0, len(split_dataset), chunk_size), total=total, desc="Summarizing"):
        batch = dict(split_dataset[i : i + chunk_size])
        n_prompts_total += sum(len(chunks) for chunks in batch["chunks"])
        result, n_truncated = summarize_batch(
            batch, prompt_template, llm, sampling_params, max_prompt_tokens
        )
        n_truncated_total += n_truncated
        subject_ids.extend(result["subject_id"])
        chunks_col.extend(result["chunks"])

    log.info(
        f"Prompt truncation: {n_truncated_total}/{n_prompts_total} prompts truncated "
        f"to {max_prompt_tokens} tokens"
    )
    return datasets.Dataset.from_dict({"subject_id": subject_ids, "chunks": chunks_col})


def chunk_text(text: str, window_regex: str) -> list[str]:
    """Split a patient EHR string into windows at each ``window_regex`` match.

    Chunks start at a match (or at the start of the string for any preamble) and
    run until the next match / end of text.
    """
    pattern = re.compile(window_regex, flags=re.MULTILINE)
    starts = [match.start() for match in pattern.finditer(text)]
    if not starts:
        return [text] if text.strip() else []

    chunks: list[str] = []
    if starts[0] > 0:
        preamble = text[: starts[0]].strip()
        if preamble:
            chunks.append(preamble)

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_batch(examples: dict, window_regex: str) -> dict:
    """Chunks the batch of patients into nested chunks.

    Returns:
        dict: A dictionary with the following keys:
            - "subject_id": list[str], The subject IDs of the patients.
            - "chunks": list[list[dict]], The nested chunks for each patient.
    """
    subject_ids: list = []
    chunks_col: list[list[dict]] = []

    for subject_id, text in zip(examples["subject_id"], examples["ehr_text"], strict=True):
        subject_ids.append(subject_id)
        chunks_col.append(
            [
                {"chunk_id": chunk_id, "text": chunk}
                for chunk_id, chunk in enumerate(chunk_text(text, window_regex))
            ]
        )

    return {
        "subject_id": subject_ids,
        "chunks": chunks_col,
    }


def load_dataset(input_dir: Path, train_split: str, val_split: str) -> datasets.DatasetDict:
    log.info(f"Loading raw dataset from {input_dir}...")
    datafiles: dict[str, list[str]] = {train_split: [], val_split: []}
    for shard_path in sorted(input_dir.glob("*/*.parquet")):
        split = shard_path.parent.name
        if split not in datafiles:
            continue
        datafiles[split].append(str(shard_path))

    if not any(datafiles.values()):
        raise FileNotFoundError(
            f"No parquet shards found under {input_dir}/<split>/*.parquet "
            f"(expected splits: {train_split!r}, {val_split!r})"
        )

    dataset = datasets.load_dataset("parquet", data_files=datafiles)
    log.info(
        f"Loaded {len(dataset[train_split])} train and {len(dataset[val_split])} val patients."
    )
    return dataset


def debug_printing(
    raw_dataset: datasets.DatasetDict,
    summarized_dataset: datasets.DatasetDict,
    train_split: str,
) -> None:
    def print_text(title: str, text: str, max_chars: int = 2000, **meta: object) -> None:
        truncated = len(text) > max_chars
        display_text = text[:max_chars] + ("\n... [truncated]" if truncated else "")
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
        for key, value in meta.items():
            print(f"{key}: {value}")
        print(f"\ntext ({len(text)} chars):\n{'-' * 80}\n{display_text}\n")

    pt_index = 1
    raw_example = raw_dataset[train_split][pt_index]
    subject_id = raw_example["subject_id"]
    print_text(f"Raw patient (subject_id={subject_id})", raw_example["ehr_text"], subject_id=subject_id)

    patient = summarized_dataset[train_split][pt_index]
    chunks = patient["chunks"]
    log.info(f"Patient {subject_id} has {len(chunks)} chunks")
    for chunk in chunks:
        print_text(
            f"Chunk {chunk['chunk_id']} / {len(chunks) - 1} (subject_id={subject_id})",
            chunk["text"],
            subject_id=subject_id,
            chunk_id=chunk["chunk_id"],
            summary=chunk.get("summary", "<missing>"),
        )


@hydra.main(version_base=None, config_path="config/summarize", config_name="summarize")
def main(cfg: DictConfig) -> None:
    """Chunk patient EHR text by date windows and summarize each chunk."""
    ### Setup
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val
    window_regex = cfg.summarization.window_regex
    prompt_template = cfg.summarization.prompt
    out_path = output_dir / "summarized_dataset"

    if not cfg.overwrite and out_path.exists():
        log.info(
            f"Summarized dataset already exists at {out_path} "
            f"and overwrite is {cfg.overwrite}. Skipping."
        )
        return

    ### Load dataset
    log.info("Loading raw pretraining dataset...")
    dataset = load_dataset(input_dir, train_split, val_split)

    max_patients = cfg.get("max_patients", None)
    if max_patients is not None:
        max_patients = int(max_patients)
        dataset = datasets.DatasetDict(
            {
                split: ds.select(range(min(max_patients, len(ds))))
                for split, ds in dataset.items()
            }
        )
        log.info(
            f"Debug subset: using first {max_patients} patients per split "
            f"({len(dataset[train_split])} train / {len(dataset[val_split])} val)."
        )

    ### Chunk patients
    log.info(f"Chunking patients with window_regex={window_regex!r}...")
    chunked_dataset = dataset.map(
        lambda batch: chunk_batch(batch, window_regex),
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        remove_columns=dataset[train_split].column_names,
    )
    log.info(
        f"Kept {len(chunked_dataset[train_split])} train / {len(chunked_dataset[val_split])} val patients "
        f"(each with a nested chunks list)."
    )

    ### Load model
    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))
    log.info(f"Loading model ({cfg.model.model})...")
    log.info(sampling_params)
    llm = LLM(**cfg.model)

    max_prompt_tokens = int(cfg.model.max_model_len) - int(cfg.sampling_params.max_tokens)
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"sampling_params.max_tokens ({cfg.sampling_params.max_tokens}) must be "
            f"< model.max_model_len ({cfg.model.max_model_len})."
        )
    log.info(f"Truncating prompts to {max_prompt_tokens} tokens when needed (keeping the end).")

    log.info("Generating summaries...")
    summarized_dataset = datasets.DatasetDict(
        {
            split: summarize_split(
                chunked_dataset[split],
                prompt_template,
                llm,
                sampling_params,
                cfg.chunk_size,
                max_prompt_tokens,
            )
            for split in chunked_dataset
        }
    )
    log.info(
        f"Summarized {len(summarized_dataset[train_split])} train and "
        f"{len(summarized_dataset[val_split])} val patients."
    )

    # summarized_dataset.save_to_disk(out_path)
    # log.info(f"Saved summarized dataset to {out_path}")

    ### Debug printing
    if cfg.get("debug", False):
        debug_printing(dataset, summarized_dataset, train_split)


if __name__ == "__main__":
    main()
