import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="merge_peft_adaptor")
def main(cfg: DictConfig) -> None:
    base_model_name_or_path = cfg.base_model
    lora_adaptor = cfg.lora_adaptor
    out_dir = Path(cfg.out_dir)
    log.info(f"Loading base model: {base_model_name_or_path} on {cfg.device_map} with dtype {cfg.dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map=cfg.device_map,
        torch_dtype=cfg.dtype,
    )

    log.info(f"Loading LoRA adaptor: {lora_adaptor}")
    model = PeftModel.from_pretrained(base_model, lora_adaptor)

    log.info("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    log.info(f"Saving merged model to {out_dir}")
    merged_model.save_pretrained(str(out_dir))

    log.info(f"Saving tokenizer to {out_dir}")
    # Try loading tokenizer from adaptor first, fallback to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_adaptor)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(str(out_dir))

    log.info("Done.")


if __name__ == "__main__":
    main()
