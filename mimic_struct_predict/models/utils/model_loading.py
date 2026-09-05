"""Model and tokeniser loading utilities."""

import logging
import os
from typing import Type

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype_str: str | None) -> torch.dtype | None:
    """Resolve a string dtype name to a ``torch.dtype``.

    Args:
        dtype_str: One of ``"float16"``, ``"bfloat16"``, ``"float32"``, or ``None``.
            When ``None``, returns ``None`` to signal auto-detection.

    Raises:
        ValueError: If ``dtype_str`` is non-``None`` but not a recognised dtype name.
    """
    if dtype_str is None:
        return None
    if dtype_str in _DTYPE_MAP:
        return _DTYPE_MAP[dtype_str]
    raise ValueError(f"Unrecognised dtype '{dtype_str}'. Expected one of {list(_DTYPE_MAP)} or null.")


def get_token_id(tokenizer: PreTrainedTokenizer, string: str) -> int:
    """Get the token ID for a given string, validating it is known and encodes to one token."""
    token_ids = tokenizer.encode(string, add_special_tokens=False)
    # 1. Check for multi-token decomposition
    if len(token_ids) != 1:
        raise ValueError(
            f"'{string}' must encode to exactly 1 token. Instead, it encoded to {len(token_ids)} tokens: {token_ids}"
        )

    token_id = token_ids[0]

    # 2. Check if it's an Out-of-Vocabulary (Unknown) token
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"'{string}' is not in the tokenizer vocabulary (UNK).")

    return int(token_id)


def get_device() -> torch.device:
    """Get the best available device for inference.

    Returns:
        torch.device: CUDA if available, then MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(tokenizer_dir: str, padding_side: str = "left") -> PreTrainedTokenizer:
    """Load a HuggingFace tokenizer from a directory.

    Args:
        tokenizer_dir: Path to the directory containing the tokenizer files.
        padding_side: Side on which to pad sequences.
    Returns:
        PreTrainedTokenizer: The loaded tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_class: Type[AutoModel],
    model_config: DictConfig,
    eval_mode: bool = False,
    padding_side: str = "left",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokeniser from a Hydra model config.

    Args:
        model_class: HuggingFace model class to instantiate
            (e.g. ``AutoModelForCausalLM``).
        model_config: Hydra ``DictConfig`` with the following keys:
            - ``pretrained_model_name_or_path``: local path or HF hub ID.
            - ``dtype``: one of ``"float16"``, ``"bfloat16"``, ``"float32"``, or
              ``null`` (auto-detect).
            - ``load_in_4bit`` / ``load_in_8bit``: enable bitsandbytes quantisation
              (CUDA only).
            - ``compile_model``: whether to run ``torch.compile``.
            - ``adapter_path``: optional path to a PEFT/LoRA adapter.
        eval_mode: If ``True``, puts the model into ``.eval()`` mode.
            If ``False`` (default), puts the model into ``.train()`` mode.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    logger.info(f"Loading model from {model_config.pretrained_model_name_or_path}")
    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = load_tokenizer(model_config.pretrained_model_name_or_path, padding_side=padding_side)

    # Determine best dtype
    resolved_dtype = resolve_dtype(model_config.dtype)
    if resolved_dtype is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            resolved_dtype = torch.bfloat16
        elif device.type in ("cuda", "mps"):
            resolved_dtype = torch.float16
        else:
            resolved_dtype = torch.float32

    # Configure quantisation if requested
    quantization_config = None
    if getattr(model_config, "load_in_4bit", False) or getattr(model_config, "load_in_8bit", False):
        if getattr(model_config, "load_in_4bit", False) and getattr(model_config, "load_in_8bit", False):
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit. Choose one.")
        if device.type != "cuda":
            logger.warning("Quantisation requires CUDA. Falling back to standard loading.")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=model_config.load_in_4bit,
                load_in_8bit=model_config.load_in_8bit,
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(
                f"Using {'4-bit' if model_config.load_in_4bit else '8-bit'} "
                f"quantisation with compute dtype {resolved_dtype}"
            )

    local_rank = os.environ.get("LOCAL_RANK")

    target_device_map: dict[str, int] | str | None
    if local_rank is not None:
        # Strict mapping for DDP: force this process to its assigned GPU
        target_device_map = {"": int(local_rank)}
        logger.info(f"DDP detected. Forcing model to cuda:{local_rank}")
    else:
        # Fallback for single-GPU or standard evaluation scripts
        target_device_map = "auto" if device.type == "cuda" else None

    model = model_class.from_pretrained(
        model_config.pretrained_model_name_or_path,
        torch_dtype=resolved_dtype,
        device_map=target_device_map,
        quantization_config=quantization_config,
        attn_implementation=getattr(model_config, "attn_implementation", None),
    )

    if hasattr(model.config, "text_config"):
        model.config.pad_token_id = (
            model.config.text_config.pad_token_id
            if model.config.text_config.pad_token_id is not None
            else model.config.text_config.eos_token_id
        )
    elif model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    adapter_path = getattr(model_config, "adapter_path", None)
    if adapter_path is not None:
        logger.info(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    if device.type != "cuda":
        model = model.to(device)

    if getattr(model_config, "compile_model", False):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.eval() if eval_mode else model.train()

    return model, tokenizer
