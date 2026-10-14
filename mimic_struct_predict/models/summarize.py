import logging
import re
from functools import partial
from pathlib import Path

import datasets
import hydra
import polars as pl
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


def format_prompt(
    prompt_template: str,
    tokenizer,
    use_chat_template: bool,
    system_prompt: str | None,
    **fields: str,
) -> str:
    """Build a generation prompt, using the instruct chat template when available."""
    user_content = prompt_template.format(**fields)
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return user_content

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as e:
        log.warning(f"apply_chat_template failed ({e}); falling back to raw prompt.")
        return user_content


def format_summarize_prompt(
    text: str,
    prompt_template: str,
    tokenizer,
    use_chat_template: bool,
    system_prompt: str | None,
) -> str:
    return format_prompt(
        prompt_template,
        tokenizer,
        use_chat_template,
        system_prompt,
        text=text,
    )


NO_PRIOR_DELTAS = "None."


def fit_delta_inputs(
    text: str,
    previous_deltas: str,
    tokenizer,
    max_content_tokens: int,
) -> tuple[str, str, bool]:
    """Fit the new excerpt and prior updates into the content token budget.

    When both do not fit, keep the end of each: the latest part of the new
    excerpt and the most recent prior updates.
    """
    prev_ids = tokenizer.encode(previous_deltas, add_special_tokens=False)
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(prev_ids) + len(text_ids) <= max_content_tokens:
        return text, previous_deltas, False

    has_history = previous_deltas.strip() not in {"", NO_PRIOR_DELTAS}
    if not has_history:
        prev_budget = min(len(prev_ids), max(0, max_content_tokens - 1))
        text_budget = max_content_tokens - prev_budget
    else:
        # Start from an even split, then give any unused share back to the longer side.
        text_budget = min(len(text_ids), max(1, max_content_tokens // 2))
        prev_budget = min(len(prev_ids), max_content_tokens - text_budget)
        text_budget = min(len(text_ids), max_content_tokens - prev_budget)

    fitted_prev = (
        tokenizer.decode(prev_ids[-prev_budget:], skip_special_tokens=True)
        if prev_budget
        else NO_PRIOR_DELTAS
    )
    fitted_text = tokenizer.decode(text_ids[-text_budget:], skip_special_tokens=True)
    return fitted_text, fitted_prev, True


def delta_summarize_batch(
    batch: dict,
    prompt_template: str,
    llm: LLM,
    sampling_params: SamplingParams,
    max_prompt_tokens: int,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
    initial_prompt: str | None = None,
    initial_system_prompt: str | None = None,
) -> tuple[dict, int]:
    """Summarize each chunk as the change since earlier clinical updates.

    The first chunk of each patient is a full summary (``initial_prompt``), so
    later chunks have a clinical picture to compare against. Later prompts see
    the summaries already produced for that patient, and those prior updates
    are truncated from the front when they exceed the prompt budget. Patients
    in the batch are generated together at each chunk index.

    Returns the updated batch and the number of prompts that needed truncation.
    """
    if not initial_prompt:
        raise ValueError("delta_summarize requires summarization.initial_prompt for the first chunk.")

    tokenizer = llm.get_tokenizer()

    def content_budget(template: str, system: str | None, **fields: str) -> int:
        wrapper_tokens = len(
            tokenizer.encode(
                format_prompt(template, tokenizer, use_chat_template, system, **fields),
                add_special_tokens=False,
            )
        )
        return max(1, max_prompt_tokens - wrapper_tokens)

    initial_text_tokens = content_budget(initial_prompt, initial_system_prompt, text="")
    max_content_tokens = content_budget(
        prompt_template, system_prompt, text="", previous_deltas=""
    )

    out_chunks: list[list[dict]] = [[dict(chunk) for chunk in chunks] for chunks in batch["chunks"]]
    histories: list[list[str]] = [[] for _ in out_chunks]
    n_truncated = 0
    max_chunks = max((len(chunks) for chunks in out_chunks), default=0)

    for c_i in range(max_chunks):
        prompts: list[str] = []
        patient_indexes: list[int] = []
        truncated_flags: list[bool] = []

        for p_i, chunks in enumerate(out_chunks):
            if c_i >= len(chunks):
                continue
            chunk = chunks[c_i]
            if not histories[p_i]:
                text, was_truncated, n_tokens = truncate_chunk_text(
                    chunk["text"], tokenizer, initial_text_tokens
                )
                if was_truncated:
                    log.debug(
                        f"Truncated initial summary for subject_id={batch['subject_id'][p_i]} "
                        f"chunk_id={chunk.get('chunk_id', c_i)}: {n_tokens} -> {initial_text_tokens} tokens"
                    )
                prompts.append(
                    format_summarize_prompt(
                        text, initial_prompt, tokenizer, use_chat_template, initial_system_prompt
                    )
                )
            else:
                previous_deltas = "\n".join(histories[p_i])
                text, previous_deltas, was_truncated = fit_delta_inputs(
                    chunk["text"],
                    previous_deltas,
                    tokenizer,
                    max_content_tokens,
                )
                if was_truncated:
                    log.debug(
                        f"Truncated delta prompt for subject_id={batch['subject_id'][p_i]} "
                        f"chunk_id={chunk.get('chunk_id', c_i)} to {max_content_tokens} content tokens"
                    )
                prompts.append(
                    format_prompt(
                        prompt_template,
                        tokenizer,
                        use_chat_template,
                        system_prompt,
                        text=text,
                        previous_deltas=previous_deltas,
                    )
                )
            if was_truncated:
                n_truncated += 1
            patient_indexes.append(p_i)
            truncated_flags.append(was_truncated)

        summaries = generate_summaries(llm, prompts, sampling_params)
        for p_i, summary, was_truncated in zip(patient_indexes, summaries, truncated_flags, strict=True):
            summary = summary.strip()
            out_chunks[p_i][c_i]["summary"] = summary
            out_chunks[p_i][c_i]["truncated"] = was_truncated
            histories[p_i].append(summary)

    return {"subject_id": batch["subject_id"], "chunks": out_chunks}, n_truncated


def summarize_batch(
    batch: dict,
    prompt_template: str,
    llm: LLM,
    sampling_params: SamplingParams,
    max_prompt_tokens: int,
    use_chat_template: bool = True,
    system_prompt: str | None = None,
) -> tuple[dict, int]:
    """Add a ``summary`` field to each nested chunk for a batch of patients.

    Returns the updated batch and the number of prompts that needed truncation.
    """
    tokenizer = llm.get_tokenizer()
    # Reserve space for the instruction wrapper / chat template around {text}.
    wrapper_tokens = len(
        tokenizer.encode(
            format_summarize_prompt("", prompt_template, tokenizer, use_chat_template, system_prompt),
            add_special_tokens=False,
        )
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
            prompts.append(
                format_summarize_prompt(
                    text, prompt_template, tokenizer, use_chat_template, system_prompt
                )
            )
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
    system_prompt = cfg.summarization.get("system_prompt", None)
    use_chat_template = bool(cfg.summarization.get("use_chat_template", True))
    mode = str(cfg.summarization.get("mode", "summarize"))
    if mode == "summarize":
        summarize_fn = summarize_batch
    elif mode == "delta_summarize":
        initial_prompt = cfg.summarization.get("initial_prompt", None)
        if not initial_prompt:
            raise ValueError(
                "summarization.initial_prompt is required when summarization.mode is delta_summarize."
            )
        initial_system_prompt = cfg.summarization.get("initial_system_prompt", None)
        summarize_fn = partial(
            delta_summarize_batch,
            initial_prompt=str(initial_prompt),
            initial_system_prompt=None if initial_system_prompt is None else str(initial_system_prompt),
        )
    else:
        raise ValueError(
            f"Unknown summarization.mode {mode!r}. Expected 'summarize' or 'delta_summarize'."
        )
    out_path = output_dir / "summarized_dataset"

    if not cfg.overwrite and out_path.exists():
        log.info(
            f"Summarized dataset already exists at {out_path} "
            f"and overwrite is {cfg.overwrite}. Skipping."
        )
        return

    ### Load dataset
    log.info("Loading raw pretraining dataset...")
    max_patients = cfg.get("max_patients", None)
    if max_patients is not None:
        max_patients = int(max_patients)
    dataset = load_dataset(input_dir, train_split, val_split, max_patients=max_patients)

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

    log.info(f"Generating summaries (mode={mode})...")
    summarized_dataset = datasets.DatasetDict(
        {
            split: summarize_split(
                chunked_dataset[split],
                prompt_template,
                llm,
                sampling_params,
                cfg.chunk_size,
                max_prompt_tokens,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
                summarize_fn=summarize_fn,
            )
            for split in chunked_dataset
        }
    )
    log.info(
        f"Summarized {len(summarized_dataset[train_split])} train and "
        f"{len(summarized_dataset[val_split])} val patients."
    )

    summarized_dataset.save_to_disk(out_path)
    log.info(f"Saved summarized dataset to {out_path}")

    ### Debug printing
    if cfg.get("debug", False):
        debug_printing(dataset, summarized_dataset, train_split)


if __name__ == "__main__":
    main()
