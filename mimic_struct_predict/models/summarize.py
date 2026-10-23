import logging
import re
from functools import partial
from pathlib import Path
from typing import Any

import datasets
import hydra
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from mimic_struct_predict.models.utils.generation import load_generation_backend
from mimic_struct_predict.models.utils.summarization import (
    NO_PRIOR_CONTEXT,
    PromptJob,
    content_budget,
    generate_summaries,
    iter_chunks,
    prepare_context_prompt,
    prepare_summary_prompt,
)

log = logging.getLogger(__name__)


def attach_generation(
    chunk: dict,
    delta: str,
    was_truncated: bool,
    text: str,
    previous_context: str,
) -> None:
    chunk["delta"] = delta
    chunk["truncated"] = was_truncated
    chunk["model_text"] = text
    chunk["model_previous_context"] = previous_context


def delta_from_generations(
    batch: dict,
    prompt_template: str,
    llm: Any,
    sampling_params: Any,
    max_prompt_tokens: int,
    use_chat_template: bool,
    system_prompt: str | None,
    initial_prompt: str | None,
    initial_system_prompt: str | None,
) -> tuple[dict, int]:
    if not initial_prompt:
        raise ValueError(
            "delta_only_summarize requires summarization.initial_prompt for the first chunk."
        )

    tokenizer = llm.get_tokenizer()
    initial_text_tokens = content_budget(
        tokenizer, initial_prompt, initial_system_prompt, max_prompt_tokens, use_chat_template, text=""
    )
    max_content_tokens = content_budget(
        tokenizer,
        prompt_template,
        system_prompt,
        max_prompt_tokens,
        use_chat_template,
        text="",
        previous_context="",
    )

    chunks_col = batch["chunks"]
    histories: list[list[str]] = [[] for _ in chunks_col]
    n_truncated = 0
    max_chunks = max((len(chunks) for chunks in chunks_col), default=0)

    for c_i in range(max_chunks):
        jobs: list[PromptJob] = []
        for p_i, chunks in enumerate(chunks_col):
            if c_i >= len(chunks):
                continue
            chunk = chunks[c_i]
            subject_id = batch["subject_id"][p_i]
            chunk_id = chunk.get("chunk_id", c_i)
            if not histories[p_i]:
                # First chunk, use initial prompt
                prompt, text, was_truncated = prepare_summary_prompt(
                    chunk["text"],
                    tokenizer=tokenizer,
                    max_text_tokens=initial_text_tokens,
                    prompt_template=initial_prompt,
                    use_chat_template=use_chat_template,
                    system_prompt=initial_system_prompt,
                    subject_id=subject_id,
                    chunk_id=chunk_id,
                )
                previous_context = NO_PRIOR_CONTEXT
            else:
                prompt, text, previous_context, was_truncated = prepare_context_prompt(
                    chunk["text"],
                    "\n".join(histories[p_i]),
                    tokenizer=tokenizer,
                    max_content_tokens=max_content_tokens,
                    prompt_template=prompt_template,
                    use_chat_template=use_chat_template,
                    system_prompt=system_prompt,
                    subject_id=subject_id,
                    chunk_id=chunk_id,
                )
            n_truncated += int(was_truncated)
            jobs.append(
                PromptJob(
                    p_i=p_i,
                    c_i=c_i,
                    prompt=prompt,
                    text=text,
                    previous_context=previous_context,
                    was_truncated=was_truncated,
                )
            )

        # At chunk index 0 every active patient uses the initial system prompt.
        active_system = initial_system_prompt if c_i == 0 else system_prompt
        deltas = generate_summaries(
            llm,
            [job.prompt for job in jobs],
            sampling_params,
            system_prompt=active_system,
        )
        for job, delta in zip(jobs, deltas, strict=True):
            delta = delta.strip()
            attach_generation(
                chunks_col[job.p_i][job.c_i],
                delta,
                job.was_truncated,
                job.text,
                job.previous_context,
            )
            histories[job.p_i].append(delta)

    return {"subject_id": batch["subject_id"], "chunks": chunks_col}, n_truncated


def delta_from_summaries(
    batch: dict,
    prompt_template: str,
    llm: Any,
    sampling_params: Any,
    max_prompt_tokens: int,
    use_chat_template: bool,
    system_prompt: str | None,
    log_example_prompt: bool = False,
) -> tuple[dict, int]:
    """Write deltas from precomputed per-chunk summaries; all prompts are independent."""
    tokenizer = llm.get_tokenizer()
    max_content_tokens = content_budget(
        tokenizer,
        prompt_template,
        system_prompt,
        max_prompt_tokens,
        use_chat_template,
        text="",
        previous_context="",
    )

    chunks_col = batch["chunks"]
    jobs: list[PromptJob] = []
    logged_example_prompt = False

    for p_i, c_i, chunk in iter_chunks(chunks_col):
        if c_i == 0:
            attach_generation(
                chunk,
                str(chunk["summary"]).strip(),
                False,
                str(chunk["text"]),
                NO_PRIOR_CONTEXT,
            )
            continue

        previous_context = "\n".join(
            str(prev["summary"]).strip() for prev in chunks_col[p_i][:c_i]
        )
        prompt, text, previous_context, was_truncated = prepare_context_prompt(
            str(chunk["text"]),
            previous_context,
            tokenizer=tokenizer,
            max_content_tokens=max_content_tokens,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            subject_id=batch["subject_id"][p_i],
            chunk_id=chunk.get("chunk_id", c_i),
        )
        jobs.append(
            PromptJob(
                p_i=p_i,
                c_i=c_i,
                prompt=prompt,
                text=text,
                previous_context=previous_context,
                was_truncated=was_truncated,
            )
        )
        if log_example_prompt and not logged_example_prompt:
            preview_limit = 4000
            user_preview = (
                prompt if len(prompt) <= preview_limit else prompt[:preview_limit] + "\n... [truncated for log]"
            )
            log.info(
                "Example delta_summarize prompt "
                "(subject_id=%s chunk_id=%s, user_chars=%d, previous_context_chars=%d):\n"
                "----- system -----\n%s\n----- user -----\n%s\n----- end -----",
                batch["subject_id"][p_i],
                chunk.get("chunk_id", c_i),
                len(prompt),
                len(previous_context),
                system_prompt or "",
                user_preview,
            )
            logged_example_prompt = True

    deltas = generate_summaries(
        llm,
        [job.prompt for job in jobs],
        sampling_params,
        system_prompt=system_prompt,
    )
    for job, delta in zip(jobs, deltas, strict=True):
        attach_generation(
            chunks_col[job.p_i][job.c_i],
            delta.strip(),
            job.was_truncated,
            job.text,
            job.previous_context,
        )

    return {"subject_id": batch["subject_id"], "chunks": chunks_col}, sum(
        job.was_truncated for job in jobs
    )


def summarize_batch(
    batch: dict,
    prompt_template: str,
    llm: Any,
    sampling_params: Any,
    max_prompt_tokens: int,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> tuple[dict, int]:
    """Add a ``summary`` field to each nested chunk for a batch of patients.

    Returns the updated batch and the number of prompts that needed truncation.
    """
    tokenizer = llm.get_tokenizer()
    max_text_tokens = content_budget(
        tokenizer, prompt_template, system_prompt, max_prompt_tokens, use_chat_template, text=""
    )

    chunks_col = batch["chunks"]
    jobs: list[PromptJob] = []
    for p_i, c_i, chunk in iter_chunks(chunks_col):
        prompt, text, was_truncated = prepare_summary_prompt(
            chunk["text"],
            tokenizer=tokenizer,
            max_text_tokens=max_text_tokens,
            prompt_template=prompt_template,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            subject_id=batch["subject_id"][p_i],
            chunk_id=chunk.get("chunk_id", c_i),
        )
        jobs.append(
            PromptJob(
                p_i=p_i,
                c_i=c_i,
                prompt=prompt,
                text=text,
                was_truncated=was_truncated,
            )
        )

    summaries = generate_summaries(
        llm,
        [job.prompt for job in jobs],
        sampling_params,
        system_prompt=system_prompt,
    )
    for job, summary in zip(jobs, summaries, strict=True):
        chunks_col[job.p_i][job.c_i]["summary"] = summary
        chunks_col[job.p_i][job.c_i]["truncated"] = job.was_truncated

    return {"subject_id": batch["subject_id"], "chunks": chunks_col}, sum(
        job.was_truncated for job in jobs
    )


def summarize_split(
    split_dataset: datasets.Dataset,
    prompt_template: str,
    llm: Any,
    sampling_params: Any,
    chunk_size: int,
    max_prompt_tokens: int,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
    summarize_fn=summarize_batch,
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
        result, n_truncated = summarize_fn(
            batch,
            prompt_template,
            llm,
            sampling_params,
            max_prompt_tokens,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
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


def load_dataset(
    input_dir: Path,
    train_split: str,
    val_split: str,
    max_patients: int | None = None,
) -> datasets.DatasetDict:
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

    if max_patients is not None:
        # Read only the first rows of the first shard so debug runs don't scan the full corpus.
        limited: dict[str, datasets.Dataset] = {}
        for split, files in datafiles.items():
            if not files:
                continue
            frame = pl.read_parquet(files[0], n_rows=max_patients)
            limited[split] = datasets.Dataset.from_dict(frame.to_dict(as_series=False))
            log.info(f"Loaded {len(limited[split])} {split} patients from {files[0]} (max_patients={max_patients}).")
        return datasets.DatasetDict(limited)

    dataset = datasets.load_dataset("parquet", data_files=datafiles)
    log.info(
        f"Loaded {len(dataset[train_split])} train and {len(dataset[val_split])} val patients."
    )
    return dataset


def load_summarized_dataset(
    input_dir: Path,
    max_patients: int | None = None,
) -> datasets.DatasetDict:
    """Load a ``summarized_dataset`` saved by a prior ``summarize`` run."""
    log.info(f"Loading summarized dataset from {input_dir}...")
    dataset = datasets.load_from_disk(str(input_dir))
    if isinstance(dataset, datasets.Dataset):
        dataset = datasets.DatasetDict({"train": dataset})
    if max_patients is not None:
        dataset = datasets.DatasetDict(
            {split: rows.select(range(min(max_patients, len(rows)))) for split, rows in dataset.items()}
        )
    for split, rows in dataset.items():
        log.info(f"Loaded {len(rows)} {split} patients from summarized dataset.")
    return dataset


### TODO: Clean up a bit when we know more about what we're doing
def build_summarize_fn(cfg: DictConfig, mode: str):
    """Select the batch function for ``mode``."""
    if mode == "summarize":
        return summarize_batch
    if mode == "delta_only_summarize":
        initial_prompt = cfg.summarization.get("initial_prompt", None)
        if not initial_prompt:
            raise ValueError(
                "summarization.initial_prompt is required when summarization.mode is "
                "delta_only_summarize."
            )
        initial_system_prompt = cfg.summarization.get("initial_system_prompt", None)
        return partial(
            delta_from_generations,
            initial_prompt=str(initial_prompt),
            initial_system_prompt=None if initial_system_prompt is None else str(initial_system_prompt),
        )
    if mode == "delta_summarize":
        return partial(
            delta_from_summaries,
            log_example_prompt=bool(cfg.summarization.get("log_example_prompt", False)),
        )
    raise ValueError(
        f"Unknown summarization.mode {mode!r}. "
        "Expected 'summarize', 'delta_summarize', or 'delta_only_summarize'."
    )


def load_chunked_dataset(cfg: DictConfig, mode: str) -> datasets.DatasetDict:
    """Load either raw EHR text (then chunk) or a prior summarized dataset."""
    input_dir = Path(cfg.input_dir)
    max_patients = cfg.get("max_patients", None)
    if max_patients is not None:
        max_patients = int(max_patients)

    if mode == "delta_summarize":
        return load_summarized_dataset(input_dir, max_patients=max_patients)

    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val
    window_regex = cfg.summarization.window_regex
    log.info("Loading raw pretraining dataset...")
    raw_dataset = load_dataset(input_dir, train_split, val_split, max_patients=max_patients)
    log.info(f"Chunking patients with window_regex={window_regex!r}...")
    chunked = raw_dataset.map(
        lambda batch: chunk_batch(batch, window_regex),
        batched=True,
        batch_size=cfg.batch_size,
        num_proc=cfg.num_workers,
        remove_columns=raw_dataset[train_split].column_names,
    )
    log.info(
        f"Kept {len(chunked[train_split])} train / {len(chunked[val_split])} val patients "
        f"(each with a nested chunks list)."
    )
    return chunked


def generate_dataset(
    chunked_dataset: datasets.DatasetDict,
    cfg: DictConfig,
    summarize_fn,
    llm: Any,
    sampling_params: Any,
    max_prompt_tokens: int,
) -> datasets.DatasetDict:
    """Run summarization over every split."""
    mode = str(cfg.summarization.get("mode", "summarize"))
    prompt_template = cfg.summarization.prompt
    system_prompt = cfg.summarization.get("system_prompt", None)
    use_chat_template = bool(cfg.summarization.get("use_chat_template", True))

    log.info(f"Generating summaries (mode={mode})...")
    return datasets.DatasetDict(
        {
            split: summarize_split(
                rows,
                prompt_template,
                llm,
                sampling_params,
                cfg.chunk_size,
                max_prompt_tokens,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                summarize_fn=summarize_fn,
            )
            for split, rows in chunked_dataset.items()
        }
    )


@hydra.main(version_base=None, config_path="config/summarize", config_name="summarize")
def main(cfg: DictConfig) -> None:
    """Chunk patient EHR text by date windows and summarize each chunk."""
    mode = str(cfg.summarization.get("mode", "summarize"))
    out_path = Path(cfg.output_dir) / "summarized_dataset"
    if not cfg.overwrite and out_path.exists():
        log.info(
            f"Summarized dataset already exists at {out_path} "
            f"and overwrite is {cfg.overwrite}. Skipping."
        )
        return

    summarize_fn = build_summarize_fn(cfg, mode)
    chunked_dataset = load_chunked_dataset(cfg, mode)
    llm, sampling_params, max_prompt_tokens = load_generation_backend(cfg)
    summarized_dataset = generate_dataset(
        chunked_dataset, cfg, summarize_fn, llm, sampling_params, max_prompt_tokens
    )

    train_split = cfg.dataset.split_names.train
    val_split = cfg.dataset.split_names.val
    if train_split not in summarized_dataset:
        train_split = next(iter(summarized_dataset))
    if val_split not in summarized_dataset:
        val_split = train_split
    log.info(
        f"Summarized {len(summarized_dataset[train_split])} train and "
        f"{len(summarized_dataset[val_split])} val patients."
    )
    summarized_dataset.save_to_disk(out_path)
    log.info(f"Saved summarized dataset to {out_path}")


if __name__ == "__main__":
    main()
