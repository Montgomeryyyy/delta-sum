"""GRPO training with custom reward functions for binary clinical prediction.

Usage:
    uv run train-grpo \
        prompt_dataset_dir=data/mimic-iv-prompts \
        output_dir=outputs/grpo
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import GPUtil
import hydra
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = get_logger(__name__)


def _extract_answer(completion: str, pattern: re.Pattern) -> str | None:
    """Extract the answer from a completion.

    Returns the answer if the completion matches the expected format,
    otherwise ``None``.
    """
    match = pattern.search(completion.strip())
    if match is None:
        return None
    return match.group(1)


def format_reward(
    completions: list[str],
    regex_pattern: re.Pattern,
    *args: Any,
    **kwargs: Any,
) -> list[float]:
    """
    0 if match
    -1 if not a match
    """
    return [-1.0 if _extract_answer(completion, regex_pattern) is None else 0.0 for completion in completions]


def calc_accuracy(
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

    If the format is invalid the reward is ``0.0``

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
        if extracted is None:
            rewards.append(0.0)
            continue

        answer = extracted == "Yes"
        rewards.append(1.0 if answer == label else 0.0)

    return rewards


def calc_brier_reward(
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

    If the format is invalid the reward is ``0``

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
            rewards.append(0.0)
            continue
        prob = float(extracted)

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

        # 4. GRPOTrainer logs eval_reward but might not return it in metrics dict
        # This prevents metric_for_best_model from using it. We pull it from log_history if missing.
        if "eval_reward" not in metrics:
            for log_entry in reversed(self.state.log_history):
                if "eval_reward" in log_entry:
                    metrics["eval_reward"] = log_entry["eval_reward"]
                    break

        return metrics


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="grpo",
)
def main(cfg: DictConfig) -> None:
    """Run GRPO training with thinking and accuracy rewards."""
    # Check the state of the current process
    state = PartialState()

    # Force non-main processes to only output warnings/errors
    if not state.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()

    dataset_dir = Path(cfg.dataset_dir)
    run_output_dir = Path(HydraConfig.get().runtime.output_dir)

    log.info(f"Model: {cfg.model}")
    log.info(f"Output directory: {run_output_dir}")

    # --- Load datasets ---
    train_dataset = datasets.load_from_disk(str(dataset_dir / cfg.split_names.train))
    log.info(
        f"Combined train dataset has {len(train_dataset)} examples and the following columns:"
        f" {train_dataset.column_names}"
    )

    eval_dict = {}
    for eval_task_dir in (dataset_dir / cfg.split_names.val).iterdir():
        if eval_task_dir.is_dir():
            full_ds = datasets.load_from_disk(str(eval_task_dir))
            if len(full_ds) > cfg.max_eval_samples:
                subsampled_ds = full_ds.shuffle(seed=cfg.seed).select(range(cfg.max_eval_samples))
                eval_dict[eval_task_dir.name] = subsampled_ds
            else:
                eval_dict[eval_task_dir.name] = full_ds

    log.info(f"Eval datasets: { {task: (len(ds), ds.column_names) for task, ds in eval_dict.items()} }")

    tracking_dict = OmegaConf.to_container(cfg.get("tracking", {}), resolve=True)
    if tracking_dict.get("report_to", "") == "tensorboard":
        os.environ["TENSORBOARD_LOGGING_DIR"] = tracking_dict.pop("logging_dir", None)

    # --- GRPO config ---
    grpo_config = GRPOConfig(
        output_dir=str(run_output_dir),
        run_name=run_output_dir.name,
        **tracking_dict,
        **OmegaConf.to_container(cfg.get("training", {}), resolve=True),
    )

    log.info(grpo_config)

    # --- LoRA config ---
    peft_config = None
    if "lora" in cfg:
        log.info("Using LoRA for training")
        peft_config = LoraConfig(**cfg.get("lora", {}))

    regex_pattern = re.compile(cfg.reward.regex_pattern)

    def _format_reward(completions: list[str], *args: Any, **kwargs: Any) -> list[float]:
        return format_reward(completions, regex_pattern, *args, **kwargs)

    if cfg.reward.output_format == "yes_no_accuracy":
        log.info("Using accuracy-based reward for training")

        def _calc_accuracy(completions: list[str], boolean_value: list[bool], *args: Any, **kwargs: Any) -> list[float]:
            return calc_accuracy(completions, boolean_value, regex_pattern, *args, **kwargs)

        reward_funcs = [_calc_accuracy, _format_reward]
    elif cfg.reward.output_format == "probability_brier":
        log.info("Using Brier score-based reward for training")

        def _calc_brier_reward(
            completions: list[str], boolean_value: list[bool], *args: Any, **kwargs: Any
        ) -> list[float]:
            return calc_brier_reward(completions, boolean_value, regex_pattern, *args, **kwargs)

        reward_funcs = [_calc_brier_reward, _format_reward]
    else:
        raise ValueError(f"Unknown reward format: {cfg.reward.output_format}")

    class DebugGRPOTrainer(MultiTaskGRPOTrainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.isfirst: bool = True
            self.issecond: bool = True

        def training_step(
            self, model: nn.Module, inputs: Union[Dict[str, Any], List[Any]], *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            if self.isfirst:
                log.info("\n--- DEBUG: Batch Check ---")
                log.info(f"Inputs Type: {type(inputs)}")

                # If it's a standard dictionary of tensors
                if isinstance(inputs, dict):
                    for key, value in inputs.items():
                        if hasattr(value, "shape"):
                            log.info(f"Key: {key:<20} Shape: {value.shape}")
                        else:
                            log.info(f"Key: {key:<20} Type: {type(value)}")

                # If it's a list (which is causing your error)
                elif isinstance(inputs, list):
                    log.info(f"Batch Size (List Length): {len(inputs)}")
                    if len(inputs) > 0:
                        log.info(f"First element type: {type(inputs[0])}")
                        # If the list contains dictionaries, let's peek at the keys
                        if isinstance(inputs[0], dict):
                            log.info(f"First element keys: {list(inputs[0].keys())}")
                            for key, value in inputs[0].items():
                                log.info(f"key {key}")
                                log.info(f"value {value}")
                                log.info("--------------------------\n")

                # --- NEW: Weight Sync Diagnostic Check ---
                if self.use_vllm:
                    log.info("\n--- DEBUG: Weight Sync Diagnostic ---")
                    try:
                        py_model = self.model
                        llm_model = self.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        vllm_names = set(p[0] for p in llm_model.named_parameters())

                        missing = []
                        synced_count = 0
                        for name, param in py_model.named_parameters():
                            # Skip the LoRA adapter weights themselves exactly as TRL does
                            if py_model.prefix in name or "original_module" in name:
                                continue

                            # Replicate TRL name-cleaning logic
                            mapped_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                            mapped_name = self.vllm_generation._fix_param_name_to_vllm(
                                mapped_name, extra_prefixes=["modules_to_save.default."]
                            )

                            if mapped_name not in vllm_names:
                                missing.append((name, mapped_name))
                            else:
                                synced_count += 1

                        log.info(f"vLLM parameter count: {len(vllm_names)}")
                        log.info(f"Parameters successfully matched: {synced_count}")
                        log.info(f"Parameters missing/unmatched in vLLM: {len(missing)}")

                        if missing:
                            log.warning(
                                "WARNING: The following parameters will fail to sync (PyTorch -> Expected vLLM):"
                            )
                            for py_name, vllm_name in missing[:15]:
                                log.warning(f"  {py_name} -> {vllm_name}")
                        else:
                            log.info("SUCCESS: All base parameter mappings successfully match vLLM weights!")
                    except Exception as e:
                        log.error(f"Failed to run weight sync diagnostic: {str(e)}")

                log.info("--------------------------\n")
                self.isfirst = False

            # Continue with the normal training step
            return super().training_step(model, inputs, *args, **kwargs)

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            if self.issecond:
                log.info(f"SELF.TEMPERATURE {self.temperature} {self.args.generation_kwargs}")
                self.issecond = False
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)

    # --- Create trainer ---
    trainer = DebugGRPOTrainer(
        model=cfg.model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dict,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
        callbacks=[MultiGpuTensorBoardCallback],
    )

    pytorch_dtype = next(trainer.model.parameters()).dtype
    log.info(f" : {pytorch_dtype}")
    # 2. Print the vLLM model dtype
    # Note: This is only directly accessible when vLLM is used in "colocate" mode.
    # Get dtype from vLLM's model config
    vllm_config_dtype = trainer.vllm_generation.llm.llm_engine.model_config.dtype
    log.info(f"vLLM config dtype: {vllm_config_dtype}")

    # Alternatively, get the actual parameter dtype from the internal vLLM model
    vllm_model = trainer.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
    vllm_param_dtype = next(vllm_model.parameters()).dtype
    log.info(f"vLLM model parameter dtype: {vllm_param_dtype}")

    # # 2. Grab the PeftModel out of the trainer
    # peft_model = trainer.model
    # print(peft_model.base_model.config._attn_implementation)
    # print(peft_model.print_trainable_parameters())
    # print("Base model dtype:", peft_model.base_model.dtype)

    # py_model = trainer.model
    # llm_model = trainer.vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model

    # log.info("py-----------------------")
    # log.info(list(name for name, _ in py_model.named_parameters()))
    # log.info("vllm---------------------")
    # log.info(list(name for name, _ in llm_model.named_parameters()))

    log.info(GPUtil.showUtilization())

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
    log.info(f"Model saved to {output_model_dir}")
