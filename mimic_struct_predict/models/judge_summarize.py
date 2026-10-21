"""Judge delta summaries using the excerpt and prior deltas the summarizer saw."""

import json
import logging
import re
from collections import Counter
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams

from mimic_struct_predict.models.utils.summarization import (
    NO_PRIOR_CONTEXT,
    content_budget,
    fit_delta_inputs,
    format_prompt,
    generate_summaries,
)

log = logging.getLogger(__name__)

SCORE_METRICS = ("coverage", "faithfulness", "novelty")
SCORE_PATTERN = re.compile(
    r"^(?:\*\*)?(COVERAGE|FAITHFULNESS|NOVELTY)(?:\*\*)?:\s*([012])\b",
    re.MULTILINE | re.IGNORECASE,
)
MAIN_ERROR_PATTERN = re.compile(
    r"^(?:\*\*)?MAIN_ERROR(?:\*\*)?:\s*([A-Za-z_]+)\b",
    re.MULTILINE | re.IGNORECASE,
)


def judgment_body(judgment: str) -> str:
    """Drop a Qwen3 reasoning block when one is present."""
    return judgment.split("</think>")[-1]


def parse_judgment_fields(judgment: str) -> dict:
    """Parse point scores and MAIN_ERROR into chunk fields for later filtering."""
    text = judgment_body(judgment)
    found = {name.lower(): int(score) for name, score in SCORE_PATTERN.findall(text)}
    main_errors = MAIN_ERROR_PATTERN.findall(text)
    return {
        "coverage": found.get("coverage"),
        "faithfulness": found.get("faithfulness"),
        "novelty": found.get("novelty"),
        "main_error": main_errors[-1].upper() if main_errors else None,
    }


def attach_judgment(chunk: dict, judgment: str) -> None:
    """Store the raw judgment and its parsed score fields on a chunk."""
    chunk["judgment"] = judgment.strip()
    chunk.update(parse_judgment_fields(chunk["judgment"]))


def aggregate_scores(dataset: datasets.DatasetDict) -> dict:
    """Score distributions and MAIN_ERROR shares across every judged chunk."""
    values = {metric: [] for metric in SCORE_METRICS}
    main_errors: list[str] = []
    n_chunks = 0
    for split in dataset.values():
        for row in split:
            for chunk in row["chunks"]:
                n_chunks += 1
                for metric in SCORE_METRICS:
                    score = chunk.get(metric)
                    if score is not None:
                        values[metric].append(int(score))
                main_error = chunk.get("main_error")
                if main_error is not None:
                    main_errors.append(str(main_error))

    metrics = {}
    for metric, scores in values.items():
        counts = Counter(scores)
        n = len(scores)
        metrics[metric.upper()] = {
            "n": n,
            "n_unparsed": n_chunks - n,
            "pct": {str(score): (100 * counts[score] / n if n else None) for score in (0, 1, 2)},
            "mean": (sum(scores) / n if n else None),
        }

    error_counts = Counter(main_errors)
    n_errors = len(main_errors)
    return {
        "n_chunks": n_chunks,
        "metrics": metrics,
        "main_errors": {
            "n": n_errors,
            "n_unparsed": n_chunks - n_errors,
            "counts": dict(error_counts.most_common()),
            "pct": {
                label: (100 * count / n_errors if n_errors else None)
                for label, count in error_counts.most_common()
            },
        },
    }


def log_score_summary(summary: dict) -> None:
    log.info(f"Aggregated scores over {summary['n_chunks']} chunks")
    for metric, stats in summary["metrics"].items():
        n = stats["n"]
        if n == 0:
            log.info(f"{metric}: no parsed scores ({stats['n_unparsed']} unparsed)")
            continue
        pct = stats["pct"]
        log.info(
            f"{metric} n={n} unparsed={stats['n_unparsed']} "
            f"2={pct['2']:.1f}% 1={pct['1']:.1f}% 0={pct['0']:.1f}% mean={stats['mean']:.3f}"
        )

    errors = summary["main_errors"]
    log.info(f"MAIN_ERROR n={errors['n']} unparsed={errors['n_unparsed']}")
    for label, count in errors["counts"].items():
        log.info(f"  {label}: {count} ({errors['pct'][label]:.1f}%)")


def model_context(chunks: list[dict]) -> list[tuple[str, str]]:
    """Return the text and previous context saved when each delta was generated."""
    return [
        (
            str(chunk["model_text"]),
            str(chunk.get("model_previous_context") or NO_PRIOR_CONTEXT),
        )
        for chunk in chunks
    ]


def chunk_delta(chunk: dict) -> str:
    """Return the generated delta text for judging."""
    return str(chunk.get("delta") or "").strip()


def judge_split(
    split: datasets.Dataset,
    prompt_template: str,
    system_prompt: str | None,
    llm: LLM,
    sampling_params: SamplingParams,
    max_prompt_tokens: int,
    use_chat_template: bool,
    batch_size: int,
) -> datasets.Dataset:
    """Add a ``judgment`` field to every chunk."""
    tokenizer = llm.get_tokenizer()
    judge_budget = content_budget(
        tokenizer,
        prompt_template,
        system_prompt,
        max_prompt_tokens,
        use_chat_template,
        text="",
        previous_context="",
        summary="",
    )

    rows = [dict(row) for row in split]
    prompts: list[str] = []
    locations: list[tuple[int, int]] = []

    for p_i, row in enumerate(rows):
        chunks = [dict(chunk) for chunk in row["chunks"]]
        row["chunks"] = chunks
        context = model_context(chunks)
        for c_i, (chunk, (text, prior)) in enumerate(zip(chunks, context, strict=True)):
            summary = chunk_delta(chunk)
            summary_tokens = len(tokenizer.encode(summary, add_special_tokens=False))
            text, prior, _ = fit_delta_inputs(
                text, prior, tokenizer, max(1, judge_budget - summary_tokens)
            )
            prompts.append(
                format_prompt(
                    prompt_template,
                    tokenizer,
                    use_chat_template,
                    system_prompt,
                    text=text,
                    previous_context=prior,
                    summary=summary,
                )
            )
            locations.append((p_i, c_i))

    judgments: list[str] = []
    for start in range(0, len(prompts), batch_size):
        judgments.extend(
            generate_summaries(llm, prompts[start : start + batch_size], sampling_params)
        )

    for (p_i, c_i), judgment in zip(locations, judgments, strict=True):
        attach_judgment(rows[p_i]["chunks"][c_i], judgment)

    return datasets.Dataset.from_list(
        [{"subject_id": row["subject_id"], "chunks": row["chunks"]} for row in rows]
    )


@hydra.main(version_base=None, config_path="config/judge", config_name="judge")
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    out_path = output_dir / "judged_dataset"
    if not cfg.overwrite and out_path.exists():
        log.info(f"Judged dataset already exists at {out_path}. Skipping.")
        return

    dataset = datasets.load_from_disk(str(input_dir))
    if isinstance(dataset, datasets.Dataset):
        dataset = datasets.DatasetDict({"train": dataset})

    max_patients = cfg.get("max_patients", None)
    if max_patients is not None:
        max_patients = int(max_patients)
        dataset = datasets.DatasetDict(
            {split: rows.select(range(min(max_patients, len(rows)))) for split, rows in dataset.items()}
        )

    sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))
    log.info(f"Loading model ({cfg.model.model})...")
    llm = LLM(**cfg.model)

    max_prompt_tokens = int(cfg.model.max_model_len) - int(cfg.sampling_params.max_tokens)
    if max_prompt_tokens <= 0:
        raise ValueError("sampling_params.max_tokens must be smaller than model.max_model_len.")

    prompt = str(cfg.judge.prompt)
    system_prompt = cfg.judge.get("system_prompt", None)
    use_chat_template = bool(cfg.judge.get("use_chat_template", True))
    judged = datasets.DatasetDict(
        {
            split: judge_split(
                rows,
                prompt,
                None if system_prompt is None else str(system_prompt),
                llm,
                sampling_params,
                max_prompt_tokens,
                use_chat_template,
                int(cfg.chunk_size),
            )
            for split, rows in dataset.items()
        }
    )
    judged.save_to_disk(out_path)
    log.info(f"Saved judged dataset to {out_path}")

    summary = aggregate_scores(judged)
    summary_path = output_dir / "score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log_score_summary(summary)
    log.info(f"Saved aggregated scores to {summary_path}")


if __name__ == "__main__":
    main()
