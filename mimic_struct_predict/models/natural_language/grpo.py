"""GRPO training with custom reward functions for binary clinical prediction.

Usage:
    uv run train-grpo \
        prompt_dataset_dir=data/mimic-iv-prompts \
        output_dir=outputs/grpo
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import datasets
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from mimic_struct_predict.models.utils.model_loading import load_model_and_tokenizer

log = logging.getLogger(__name__)


def _extract_answer(completion: str, pattern: re.Pattern) -> str | None:
    """Extract the answer from a completion.

    Returns the answer if the completion matches the expected format,
    otherwise ``None``.
    """
    match = pattern.search(completion.strip())
    if match is None:
        return None
    return match.group(1)


def calc_accuracy_and_format_reward(
    completions: list[str],
    boolean_value: list[bool],
    regex_pattern: re.Pattern,
    *args: Any,
    **kwargs: Any,
) -> list[float]:
    r"""Return per-completion rewards based on accuracy and format.

    Regex pattern should be designed to extract a "Yes" or "No" answer from the completion
    and check correctness of rest of the completion format.
    Example regex "^<think>[\s\S]*?</think>[\s\S]*?<answer>\n(Yes|No)$"

    If the format is invalid the reward is ``-1.0``

    Otherwise the reward is ``1.0`` if the answer is correct and ``0.0`` otherwise.

    Args:
        completions: Generated text completions.
        boolean_value: List of boolean values indicating the ground-truth labels.
        regex_pattern: A regex pattern with a single capturing group designed to extract the "Yes"/"No" answer
        from the completion and check correctness of rest of the completion format.
    """
    rewards = []
    for completion, label in zip(completions, boolean_value):
        extracted = _extract_answer(completion, regex_pattern)
        if extracted is None or extracted not in {"Yes", "No"}:
            rewards.append(-1.0)
            continue

        answer = extracted == "Yes"
        rewards.append(1.0 if answer == label else 0.0)

    return rewards


def calc_brier_and_format_reward(
    completions: list[str],
    boolean_value: list[bool],
    regex_pattern: re.Pattern,
    *args: Any,
    **kwargs: Any,
) -> list[float]:
    r"""Return per-completion rewards based on Brier score and format.

    Regex pattern should be designed to extract a probability from the completion as a decimal number between 0 and 1,
    and check correctness of rest of the completion format.
    Example regex: "^<think>[\s\S]*?</think>[\s\S]*?<answer>\s*?(0(?:\.\d+)?|1(?:\.0+)?)\s*?</answer>$"

    If the format is invalid the reward is ``-1.0``

    Otherwise the reward is ``1 - brier_score`` where
    ``brier_score = (probability - target)^2`` and
    ``target`` is ``1.0`` for ``"Yes"`` and ``0.0`` for ``"No"``.

    Args:
        completions: Generated text completions.
        boolean_value: Ground-truth labels (``True`` or ``False``).
    """
    rewards = []
    for completion, label in zip(completions, boolean_value):
        extracted = _extract_answer(completion, regex_pattern)
        if extracted is None:
            rewards.append(-1.0)
            continue
        try:
            prob = float(extracted)
        except ValueError:
            rewards.append(-1.0)
            continue
        if not (0.0 <= prob <= 1.0):
            rewards.append(-1.0)
            continue

        target = 1.0 if label else 0.0
        brier_score = (prob - target) ** 2
        rewards.append(1.0 - brier_score)

    return rewards


class MultiTaskGRPOTrainer(GRPOTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Any:
        # 1. Run the standard evaluation to get the task-specific metrics
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 2. Find all keys that represent a task loss
        # This will catch 'eval_icu_mortality_loss', etc.
        loss_keys = [k for k in metrics.keys() if k.endswith("_loss") and k != "eval_loss"]

        # 3. Compute the mean of those losses
        if loss_keys:
            mean_loss = sum(metrics[k] for k in loss_keys) / len(loss_keys)

            # Add it back to the metrics dictionary so load_best_model_at_end can find it
            metrics["eval_loss"] = mean_loss

            # Optional: Manually log it so it shows up in WandB/TensorBoard alongside the others
            self.log({"eval_loss": mean_loss})

        return metrics


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="grpo",
)
def main(cfg: DictConfig) -> None:
    """Run GRPO training with thinking and accuracy rewards."""
    dataset_dir = Path(cfg.dataset_dir)
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_name = cfg.model.pretrained_model_name_or_path

    log.info(f"Model: {model_name}")
    log.info(f"Output directory: {run_output_dir}")

    # --- Load model and tokeniser ---
    model, tokenizer = load_model_and_tokenizer(AutoModelForCausalLM, cfg.model, eval_mode=False, padding_side="left")

    # --- Load datasets ---
    train_dataset = datasets.load_from_disk(str(dataset_dir / cfg.split_names.train))
    train_dataset = train_dataset.select(range(1))
    log.info(
        f"Combined train dataset has {len(train_dataset)} examples and the following columns:"
        f" {train_dataset.column_names}"
    )
    eval_dict = {
        eval_task_dir.name: datasets.load_from_disk(str(eval_task_dir)).select(range(1))
        for eval_task_dir in (dataset_dir / cfg.split_names.val).iterdir()
        if eval_task_dir.is_dir()
    }
    log.info(f"Eval datasets: { {task: (len(ds), ds.column_names) for task, ds in eval_dict.items()} }")

    # --- GRPO config ---
    grpo_config = GRPOConfig(
        output_dir=str(run_output_dir),
        run_name=run_output_dir.name,
        **cfg.get("tracking", {}),
        **cfg.get("training", {}),
    )

    print(grpo_config)

    # --- LoRA config ---
    peft_config = None
    if "lora" in cfg:
        log.info("Using LoRA for training")
        peft_config = LoraConfig(**cfg.get("lora", {}))

    regex_pattern = re.compile(cfg.reward.regex_pattern)

    if cfg.reward.output_format == "yes_no_accuracy":
        log.info("Using accuracy-based reward for training")

        def _calc_accuracy_and_format_reward(
            completions: list[str], boolean_value: list[bool], *args: Any, **kwargs: Any
        ) -> list[float]:
            return calc_accuracy_and_format_reward(completions, boolean_value, regex_pattern, *args, **kwargs)

        reward_func = _calc_accuracy_and_format_reward
    elif cfg.reward.output_format == "probability_brier":
        log.info("Using Brier score-based reward for training")

        def _calc_brier_and_format_reward(
            completions: list[str], boolean_value: list[bool], *args: Any, **kwargs: Any
        ) -> list[float]:
            return calc_brier_and_format_reward(completions, boolean_value, regex_pattern, *args, **kwargs)

        reward_func = _calc_brier_and_format_reward
    else:
        raise ValueError(f"Unknown reward format: {cfg.reward.output_format}")

    # --- Create trainer ---
    trainer = MultiTaskGRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dict,
        peft_config=peft_config,
        reward_funcs=reward_func,
    )

    # --- Train ---
    log.info("Starting GRPO training...")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    final_eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", final_eval_metrics)
    trainer.save_metrics("eval", final_eval_metrics)

    # --- Save ---
    output_model_dir = run_output_dir / "final_model"
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))
    log.info(f"Model saved to {output_model_dir}")
