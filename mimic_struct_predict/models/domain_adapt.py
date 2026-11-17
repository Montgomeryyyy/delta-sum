"""Domain-adapt a causal LM to raw EHR text with LoRA, then save a merged model.

Continued pretraining (next-token loss) on ``ehr_text`` from ``input_dir/<split>/*.parquet``,
the same raw data that ``summarize.py`` reads. The merged model in ``<run_dir>/merged_model``
is a plain Hugging Face checkpoint, so it can be passed straight to vLLM as ``model.model``.
"""

import gc
import logging
import os
from itertools import chain
from pathlib import Path

import datasets
import hydra
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from mimic_struct_predict.models.utils.trainer_callbacks import MultiGpuTensorBoardCallback

log = logging.getLogger(__name__)


def load_raw_text(
    input_dir: Path, split_names: dict[str, str], max_patients: dict[str, int | None], seed: int
) -> datasets.DatasetDict:
    """Load ``ehr_text`` per split, keeping a random subset of ``max_patients[split]`` patients."""
    datafiles = {
        name: sorted(str(path) for path in (input_dir / split).glob("*.parquet"))
        for name, split in split_names.items()
    }
    missing = [split_names[name] for name, files in datafiles.items() if not files]
    if missing:
        raise FileNotFoundError(f"No parquet shards under {input_dir} for splits {missing}")

    dataset = datasets.load_dataset("parquet", data_files=datafiles, columns=["ehr_text"])
    for name, rows in dataset.items():
        limit = max_patients.get(name)
        if limit is not None and limit < len(rows):
            dataset[name] = rows.shuffle(seed=seed).select(range(limit)).flatten_indices()
        log.info(f"Using {len(dataset[name]):,} of {len(rows):,} {name} patients.")
    return dataset


def tokenize_and_pack(
    dataset: datasets.DatasetDict, tokenizer: PreTrainedTokenizerBase, block_size: int, num_proc: int
) -> datasets.DatasetDict:
    """Tokenize each patient as ``<bos> text <eos>`` and pack all patients into ``block_size`` blocks."""
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id]

    def tokenize(batch: dict) -> dict:
        ids = tokenizer(batch["ehr_text"], add_special_tokens=False)["input_ids"]
        return {"input_ids": [bos + row + eos for row in ids]}

    def pack(batch: dict) -> dict:
        flat = list(chain.from_iterable(batch["input_ids"]))
        n_full = (len(flat) // block_size) * block_size
        return {"input_ids": [flat[i : i + block_size] for i in range(0, n_full, block_size)]}

    num_proc = num_proc if num_proc > 1 else None
    tokenized = dataset.map(tokenize, batched=True, num_proc=num_proc, remove_columns=["ehr_text"])
    packed = tokenized.map(pack, batched=True, batch_size=1000, num_proc=num_proc)
    for name, rows in packed.items():
        log.info(f"{name}: {len(rows):,} blocks of {block_size} tokens ({len(rows) * block_size:,} tokens).")
    return packed


def load_model(cfg: DictConfig) -> torch.nn.Module:
    model_kwargs = OmegaConf.to_container(cfg.model.get("config", {}), resolve=True)
    if cfg.model.get("load_in_4bit", False):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    log.info(f"Loading base model {cfg.model.name} with {model_kwargs}...")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)
    if cfg.model.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(cfg.training_args.get("gradient_checkpointing", False))
        )

    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, **cfg.lora))
    trainable, total = model.get_nb_trainable_parameters()
    log.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model


def save_merged_model(base_model: str, adapter_dir: Path, out_dir: Path, dtype: str) -> None:
    """Merge the LoRA adapter into full-precision base weights and save a standalone checkpoint."""
    log.info(f"Merging adapter {adapter_dir} into {base_model} on CPU...")
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    # vLLM needs the processor files (chat template, image config) for multimodal models like MedGemma.
    try:
        AutoProcessor.from_pretrained(base_model).save_pretrained(str(out_dir))
    except (OSError, ValueError):
        AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))
    log.info(f"Merged model saved to {out_dir}")


@hydra.main(version_base=None, config_path="config/domain_adapt", config_name="adapt_medgemma")
def main(cfg: DictConfig) -> None:
    Accelerator()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        **OmegaConf.to_container(cfg.training_args, resolve=True),
    )
    if training_args.report_to and "tensorboard" in training_args.report_to:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_names = OmegaConf.to_container(cfg.split_names, resolve=True)
    max_patients = {"train": cfg.get("max_train_patients", None), "val": cfg.get("max_eval_patients", None)}
    with training_args.main_process_first(desc="tokenizing"):
        raw = load_raw_text(Path(cfg.input_dir), split_names, max_patients, int(cfg.seed))
        dataset = tokenize_and_pack(raw, tokenizer, int(cfg.block_size), int(cfg.num_workers))

    train_dataset = dataset["train"]
    eval_dataset = dataset["val"]
    if len(eval_dataset) > cfg.max_eval_blocks:
        eval_dataset = eval_dataset.shuffle(seed=cfg.seed).select(range(int(cfg.max_eval_blocks)))

    model = load_model(cfg)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[MultiGpuTensorBoardCallback()],
    )

    log.info(f"Eval before adaptation: {trainer.evaluate()}")
    train_result = trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint", None))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info(f"Adapter saved to {adapter_dir}")

    if cfg.merge and trainer.is_world_process_zero():
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        save_merged_model(cfg.model.name, adapter_dir, run_dir / "merged_model", str(cfg.model.config.dtype))


if __name__ == "__main__":
    main()
