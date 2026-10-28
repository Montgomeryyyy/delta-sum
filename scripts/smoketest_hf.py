#!/usr/bin/env python3
"""Least-viable smoke test: load an HF causal LM and run the delta-summarize prompts.

Independent of Hydra / the full summarize pipeline. Example:

  uv run python scripts/smoketest_hf.py
  uv run python scripts/smoketest_hf.py --model google/gemma-3-270m
  uv run python scripts/smoketest_hf.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INITIAL_SYSTEM = (
    "You are a clinical summarization system for longitudinal EHRs. "
    "Produce concise, factual summaries. Preserve clinically relevant information "
    "and do not invent facts."
)

INITIAL_USER = """Summarize the clinically relevant information in the EHR excerpt below.

Include when relevant:
- diagnoses, symptoms, treatments, procedures, and medications
- laboratory results, vital signs, measurements, imaging, and study findings
- meaningful worsening, improvement, resolution, or treatment response
- clinically relevant normal, stable, benign, or negative findings

Remove redundancy and administrative detail.
Do not infer information not stated in the excerpt.

EHR excerpt:
{text}

Clinical summary:"""

DELTA_SYSTEM = (
    "You are a clinical summarization system for longitudinal EHRs. "
    "Produce concise, factual clinical updates. Preserve clinically relevant new "
    "information and do not invent facts."
)

DELTA_USER = """Given the previous clinical updates and the new EHR excerpt, extract the
clinically relevant information that is not already represented in the
previous updates.

Include when relevant:
- new diagnoses, symptoms, treatments, procedures, or medications
- new laboratory results, vital signs, measurements, imaging, or study findings
- meaningful worsening, improvement, resolution, or treatment response
- new clinically relevant evidence that a known finding is stable, unchanged,
  benign, or absent

Novelty is relative to the previous updates provided below.
Do not assume that information omitted from them is already known.

Do not repeat information already represented with the same clinical meaning.
Routine repeated measurements need not be included unless they add clinically
relevant information or change the interpretation of the patient's state.
Do not include administrative details.
Do not infer information not stated in the excerpt.

Output exactly "No clinically meaningful update." only if the excerpt adds no
clinically relevant information beyond the previous updates.

Previous clinical updates:
{previous_context}

New EHR excerpt:
{text}

Clinical update:"""

SAMPLE_EHR_1 = """### Time 0h 0m
#### ED Admission
Age 61
Gender Female
Chief Complaint Left flank pain
Heart Rate bpm 102
Systolic Blood Pressure mmHg 136
"""

SAMPLE_EHR_2 = """### Time 6h 0m
#### Hospital Transfer
Transferred to floor.
Creatinine mg/dL 1.8 (up from 1.1)
Pain improved after fluids.
"""


def build_prompt(tokenizer, system: str, user: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system}\n\n{user}"


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"Loading tokenizer/model={args.model!r} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(args.device)
    model.eval()
    print(f"Loaded. params={model.num_parameters():,} device={next(model.parameters()).device}")

    # 1) Initial full summary (first window)
    initial_prompt = build_prompt(
        tokenizer,
        INITIAL_SYSTEM,
        INITIAL_USER.format(text=SAMPLE_EHR_1),
    )
    print("\n=== initial summarize ===")
    summary = generate(model, tokenizer, initial_prompt, args.max_new_tokens)
    print(summary or "<empty>")

    # 2) Delta update conditioned on prior summary
    delta_prompt = build_prompt(
        tokenizer,
        DELTA_SYSTEM,
        DELTA_USER.format(previous_context=summary or "(none)", text=SAMPLE_EHR_2),
    )
    print("\n=== delta update ===")
    delta = generate(model, tokenizer, delta_prompt, args.max_new_tokens)
    print(delta or "<empty>")
    print("\nOK: HF load + generate with summarize prompts succeeded.")


if __name__ == "__main__":
    main()
