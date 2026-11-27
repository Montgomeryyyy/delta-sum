"""Linear probing (logistic regression) on frozen pre-trained model embeddings for downstream tasks."""

import gc
import json
import logging
from pathlib import Path
from typing import Any

import datasets
import hydra
import numpy as np
import polars as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from mimic_struct_predict.models.generative_inference import truncate_ids

log = logging.getLogger(__name__)

LABEL_COLUMN = "boolean_value"


def load_eval_model(
    model_path: str | Path,
    tokenizer_dir: str | Path | None,
    model_kwargs: dict[str, Any],
    device: str,
    base_model: str | Path | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM for probing, including LoRA/PEFT Trainer checkpoints.

    If ``adapter_config.json`` is present under ``model_path``, loads the base model
    then attaches the adapter. ``base_model`` overrides the base id from the adapter
    config when set.
    """
    model_path = Path(model_path)
    adapter_config_path = model_path / "adapter_config.json"
    tok_source = str(tokenizer_dir or (base_model if adapter_config_path.exists() else model_path))
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(model_kwargs)
    if adapter_config_path.exists():
        adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_name = str(base_model or adapter_cfg.get("base_model_name_or_path"))
        if not base_name:
            raise ValueError(
                f"LoRA checkpoint at {model_path} has no base_model_name_or_path; "
                "pass base_model=... explicitly."
            )
        log.info(f"Loading base model '{base_name}' + adapter from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        log.info(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)

    model = model.to(device)
    model.eval()
    return model, tokenizer


def subsample(dataset: datasets.Dataset, max_samples: int | None, seed: int) -> datasets.Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


@torch.inference_mode()
def compute_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: list[list[int]],
    batch_size: int,
    pooling: str,
    layer: int,
    pad_to_multiple_of: int | None,
) -> np.ndarray:
    """Embed each sequence by pooling the hidden states of a given layer."""
    if tokenizer.padding_side != "left":
        raise ValueError("Tokenizer must use left padding so the final position is the last real token")

    # Batch similar lengths together to minimise padding, then restore the original order
    order = np.argsort([-len(ids) for ids in input_ids], kind="stable")
    pooled_batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(order), batch_size), desc="Embedding"):
        batch_idx = order[start : start + batch_size]
        encoded = tokenizer.pad(
            {"input_ids": [input_ids[i] for i in batch_idx]},
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**encoded, output_hidden_states=True, logits_to_keep=1)
        hidden = outputs.hidden_states[layer]

        if pooling == "last":
            pooled = hidden[:, -1]
        elif pooling == "mean":
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling '{pooling}'. Expected 'last' or 'mean'.")

        pooled_batches.append(pooled.float().cpu().numpy())

    sorted_embeddings = np.concatenate(pooled_batches)
    embeddings = np.empty_like(sorted_embeddings)
    embeddings[order] = sorted_embeddings
    return embeddings


def build_probe(probe_cfg: DictConfig, seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=list(probe_cfg.Cs),
            cv=StratifiedKFold(n_splits=probe_cfg.cv_folds, shuffle=True, random_state=seed),
            scoring="roc_auc",
            max_iter=probe_cfg.max_iter,
            n_jobs=probe_cfg.n_jobs,
            random_state=seed,
        ),
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_prob >= 0.5)),
    }


def embed_split(
    dataset: datasets.Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = cfg.embedding.get("prompt_suffix")
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"] if suffix else []
    max_len = cfg.tokenizer.max_input_length
    context_len = max_len - len(suffix_ids) if max_len is not None else None
    input_ids = [
        truncate_ids(ids, context_len, cfg.tokenizer.keep_first_token) + suffix_ids
        for ids in dataset["input_ids"]
    ]
    embeddings = compute_embeddings(
        model,
        tokenizer,
        input_ids,
        batch_size=cfg.embedding.batch_size,
        pooling=cfg.embedding.pooling,
        layer=cfg.embedding.layer,
        pad_to_multiple_of=cfg.embedding.pad_to_multiple_of,
    )
    labels = np.asarray(dataset[LABEL_COLUMN], dtype=np.int64)
    return embeddings, labels


def evaluate_task(
    task: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    task_out_dir: Path,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Embed the train/test splits of a task, fit a logistic regression probe and score it on the test split."""
    dataset_dict = datasets.load_from_disk(str(Path(cfg.task_dataset_dir) / "tokenized" / task))
    train_ds = subsample(dataset_dict[cfg.train_split], cfg.get("max_train_samples"), cfg.seed)
    log.info(f"[{task}] Embedding {len(train_ds)} '{cfg.train_split}' samples")
    x_train, y_train = embed_split(train_ds, model, tokenizer, cfg)

    test_ds = subsample(dataset_dict[cfg.test_split], cfg.get("max_test_samples"), cfg.seed)
    log.info(f"[{task}] Embedding {len(test_ds)} '{cfg.test_split}' samples")
    x_test, y_test = embed_split(test_ds, model, tokenizer, cfg)

    log.info(f"[{task}] Fitting probe on '{cfg.train_split}', evaluating on '{cfg.test_split}'")
    probe = build_probe(cfg.probe, cfg.seed)
    probe.fit(x_train, y_train)
    y_prob = probe.predict_proba(x_test)[:, 1]

    task_out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_embeddings:
        np.savez(task_out_dir / f"{cfg.train_split}_embeddings.npz", embeddings=x_train, labels=y_train)
        np.savez(task_out_dir / f"{cfg.test_split}_embeddings.npz", embeddings=x_test, labels=y_test)
    pl.DataFrame(
        {"subject_id": test_ds["subject_id"], "label": y_test, "probability": y_prob}
    ).write_parquet(task_out_dir / "predictions.parquet")

    metrics = {"task": task, "best_C": float(probe[-1].C_[0]), **compute_metrics(y_test, y_prob)}
    (task_out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info(f"[{task}] {metrics}")
    return metrics


@hydra.main(
    version_base=None,
    config_path="config/evaluate_tasks",
)
def main(cfg: DictConfig) -> None:
    """Linear-probe each pre-trained model's embedding space on each downstream task."""
    out_dir = Path(HydraConfig.get().run.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results: list[dict[str, Any]] = []
    model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)
    base_model = cfg.get("base_model")
    for model_name, model_path in cfg.models.items():
        log.info(f"Loading model '{model_name}' ({model_path}) onto {device}...")
        model, tokenizer = load_eval_model(
            model_path,
            cfg.get("tokenizer_dir"),
            model_kwargs,
            device,
            base_model=base_model,
        )

        for task in cfg.tasks:
            metrics = evaluate_task(task, model, tokenizer, out_dir / model_name / task, cfg)
            all_results.append({"model": model_name, **metrics})

        del model
        gc.collect()
        torch.cuda.empty_cache()

    results = pl.DataFrame(all_results)
    results.write_csv(out_dir / "results.csv")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        log.info(f"Results:\n{results}")
    log.info(f"Evaluation complete! Outputs saved to {out_dir}")
