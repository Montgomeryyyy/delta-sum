"""Evaluate downstream linear probes at every pretraining checkpoint and log to TensorBoard."""

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import hydra
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import set_seed

from mimic_struct_predict.models.evaluate_tasks import evaluate_task, load_eval_model

log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def list_checkpoint_dirs(train_run_dir: Path) -> list[Path]:
    ckpts = [
        path
        for path in train_run_dir.iterdir()
        if path.is_dir() and CHECKPOINT_RE.match(path.name)
    ]
    return sorted(ckpts, key=lambda path: int(CHECKPOINT_RE.match(path.name).group(1)))


def checkpoint_global_step(ckpt_dir: Path) -> int:
    state_path = ckpt_dir / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if "global_step" in state:
            return int(state["global_step"])
    match = CHECKPOINT_RE.match(ckpt_dir.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot determine global_step for {ckpt_dir}")


def checkpoint_done(probe_dir: Path, tasks: list[str]) -> bool:
    results_path = probe_dir / "results.csv"
    if not results_path.exists():
        return False
    return all((probe_dir / task / "metrics.json").exists() for task in tasks)


def log_metrics_to_tensorboard(
    writer: SummaryWriter,
    metrics: dict[str, Any],
    metric_names: list[str],
    global_step: int,
) -> None:
    task = metrics["task"]
    for name in metric_names:
        if name not in metrics:
            log.warning(f"Metric '{name}' missing for task '{task}'; skipping")
            continue
        value = metrics[name]
        if value is None or (isinstance(value, float) and value != value):  # NaN
            continue
        writer.add_scalar(f"task_probe/{task}/{name}", float(value), global_step)


def evaluate_checkpoint(
    ckpt_dir: Path,
    probe_root: Path,
    cfg: DictConfig,
    writer: SummaryWriter,
    device: str,
    tag: str | None = None,
    global_step: int | None = None,
) -> list[dict[str, Any]]:
    name = tag or ckpt_dir.name
    probe_dir = probe_root / name
    step = global_step if global_step is not None else checkpoint_global_step(ckpt_dir)
    metric_names = list(cfg.tensorboard_metrics)

    if cfg.skip_existing and checkpoint_done(probe_dir, list(cfg.tasks)):
        log.info(f"Skipping {name} (already evaluated)")
        # Re-log existing metrics so TB stays complete if events were wiped
        for task in cfg.tasks:
            metrics = json.loads((probe_dir / task / "metrics.json").read_text(encoding="utf-8"))
            log_metrics_to_tensorboard(writer, metrics, metric_names, step)
        writer.flush()
        return [{"checkpoint": name, "global_step": step, **json.loads((probe_dir / t / "metrics.json").read_text())} for t in cfg.tasks]

    log.info(f"Evaluating {name} at step {step}")
    model_kwargs = OmegaConf.to_container(cfg.model_kwargs, resolve=True)
    model, tokenizer = load_eval_model(
        ckpt_dir,
        cfg.get("tokenizer_dir"),
        model_kwargs,
        device,
        base_model=cfg.get("base_model"),
    )

    all_metrics: list[dict[str, Any]] = []
    try:
        for task in cfg.tasks:
            metrics = evaluate_task(task, model, tokenizer, probe_dir / task, cfg)
            log_metrics_to_tensorboard(writer, metrics, metric_names, step)
            all_metrics.append({"checkpoint": name, "global_step": step, **metrics})
        writer.flush()
        pl.DataFrame(all_metrics).write_csv(probe_dir / "results.csv")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return all_metrics


def run_pass(cfg: DictConfig, writer: SummaryWriter, device: str) -> list[dict[str, Any]]:
    train_run_dir = Path(cfg.train_run_dir)
    probe_root = Path(cfg.output_dir)
    probe_root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for ckpt_dir in list_checkpoint_dirs(train_run_dir):
        all_results.extend(evaluate_checkpoint(ckpt_dir, probe_root, cfg, writer, device))

    final_dir = train_run_dir / "final_model"
    if final_dir.is_dir():
        # Prefer trainer_state in the latest checkpoint for step; else last ckpt step.
        ckpts = list_checkpoint_dirs(train_run_dir)
        step = checkpoint_global_step(ckpts[-1]) if ckpts else 0
        all_results.extend(
            evaluate_checkpoint(
                final_dir,
                probe_root,
                cfg,
                writer,
                device,
                tag="final_model",
                global_step=step,
            )
        )
    return all_results


@hydra.main(version_base=None, config_path="config", config_name="evaluate_checkpoints/natural_language")
def main(cfg: DictConfig) -> None:
    train_run_dir = Path(cfg.train_run_dir)
    if not train_run_dir.is_dir():
        raise FileNotFoundError(f"train_run_dir not found: {train_run_dir}")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_root = Path(cfg.output_dir)
    # Keep sibling *_tb next to probe root (default: task_probe -> task_probe_tb)
    tb_dir = Path(cfg.get("tensorboard_dir") or f"{probe_root}_tb")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    log.info(f"Sweeping checkpoints under {train_run_dir}")
    log.info(f"Probe output dir: {probe_root}")
    log.info(f"TensorBoard dir: {tb_dir} (open parent run dir in TensorBoard)")
    log.info(f"Logging metrics: {list(cfg.tensorboard_metrics)}")

    poll_seconds = int(cfg.poll_seconds)
    max_poll_hours = float(cfg.max_poll_hours)
    deadline = time.time() + max_poll_hours * 3600 if poll_seconds > 0 else None
    all_results: list[dict[str, Any]] = []

    try:
        while True:
            all_results = run_pass(cfg, writer, device)
            final_exists = (train_run_dir / "final_model").is_dir()
            if poll_seconds <= 0 or final_exists:
                if final_exists:
                    log.info("final_model present; finishing sweep")
                break
            if deadline is not None and time.time() >= deadline:
                log.warning(f"Reached max_poll_hours={max_poll_hours}; stopping")
                break
            log.info(f"Sleeping {poll_seconds}s before next checkpoint poll...")
            time.sleep(poll_seconds)
    finally:
        writer.close()

    if all_results:
        summary = pl.DataFrame(all_results)
        summary_path = probe_root / "all_results.csv"
        summary.write_csv(summary_path)
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            log.info(f"Results:\n{summary}")
        log.info(f"Wrote {summary_path}")
    log.info("Checkpoint evaluation complete")
