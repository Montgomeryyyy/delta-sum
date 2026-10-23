"""Local vLLM and remote ARC AI generation backends for summarization."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

import requests
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class GenerationBackend(Protocol):
    """Minimal interface shared by local and remote summarization backends."""

    def get_tokenizer(self) -> Any: ...

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        system_prompt: str | None = None,
    ) -> list[str]: ...


class NoTruncateTokenizer:
    """Tokenizer stub that disables local truncation for remote long-context APIs.

    ``encode`` always returns an empty id list, so the existing truncate helpers
    treat every input as already within budget.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return []

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return ""


class LocalBackend:
    """Local model served with vLLM."""

    def __init__(self, llm: Any):
        self._llm = llm

    def get_tokenizer(self) -> Any:
        return self._llm.get_tokenizer()

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        system_prompt: str | None = None,
    ) -> list[str]:
        del system_prompt  # already applied via chat template when formatting prompts
        if not prompts:
            return []
        outputs = self._llm.generate(prompts, sampling_params=sampling_params)
        return [output.outputs[0].text for output in outputs]


class ArcBackend:
    """Remote ARC AI chat-completions client (``ai.create.kcl.ac.uk``)."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        tokenizer_name: str | None = None,
        max_concurrency: int = 8,
        timeout_s: float = 120.0,
        max_retries: int = 5,
        request_interval_s: float = 0.0,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_concurrency = max(1, int(max_concurrency))
        self.timeout_s = float(timeout_s)
        self.max_retries = max(1, int(max_retries))
        self.request_interval_s = max(0.0, float(request_interval_s))
        self.pace_lock = threading.Lock()
        self.next_request_time = 0.0
        if tokenizer_name:
            from transformers import AutoTokenizer

            log.info("Loading optional truncation tokenizer: %s", tokenizer_name)
            self.tokenizer: Any = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
        else:
            log.info("No tokenizer configured for ARC backend; local truncation disabled.")
            self.tokenizer = NoTruncateTokenizer()

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def wait_for_pace(self) -> None:
        """Sleep so successive ARC calls are at least ``request_interval_s`` apart."""
        if self.request_interval_s <= 0:
            return
        with self.pace_lock:
            now = time.monotonic()
            wait_s = self.next_request_time - now
            if wait_s > 0:
                time.sleep(wait_s)
            self.next_request_time = time.monotonic() + self.request_interval_s

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any,
        *,
        system_prompt: str | None = None,
    ) -> list[str]:
        if not prompts:
            return []
        if self.max_concurrency == 1 or len(prompts) == 1:
            return [
                self.chat_completion(prompt, sampling_params, system_prompt=system_prompt)
                for prompt in prompts
            ]

        results: list[str | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(prompts))) as pool:
            futures = {
                pool.submit(
                    self.chat_completion, prompt, sampling_params, system_prompt=system_prompt
                ): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [text if text is not None else "" for text in results]

    def chat_completion(
        self,
        prompt: str,
        sampling_params: Any,
        *,
        system_prompt: str | None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(sampling_params.max_tokens),
            "temperature": float(sampling_params.temperature),
        }
        if sampling_params.get("seed", None) is not None:
            body["seed"] = int(sampling_params.seed)

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        transient_statuses = {408, 429, 500, 502, 503, 504}
        attempt = 0
        while True:
            attempt += 1
            try:
                self.wait_for_pace()
                response = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
                if response.status_code in transient_statuses:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None and str(retry_after).isdigit():
                        sleep_s = float(retry_after)
                    else:
                        sleep_s = min(60.0, 2.0 ** min(attempt - 1, 5))
                    log.warning(
                        "ARC transient error %s (attempt %s); retrying in %.1fs",
                        response.status_code,
                        attempt,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                response.raise_for_status()
                payload = response.json()
                return str(payload["choices"][0]["message"]["content"])
            except requests.HTTPError:
                raise
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"ARC chat completion failed after {self.max_retries} attempts"
                    ) from exc
                sleep_s = min(60.0, 2.0 ** min(attempt - 1, 5))
                log.warning(
                    "ARC request failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    sleep_s,
                )
                time.sleep(sleep_s)


def load_generation_backend(cfg: DictConfig) -> tuple[GenerationBackend, Any, int]:
    """Construct the configured backend and return ``(backend, sampling_params, max_prompt_tokens)``."""
    model_cfg = cfg.model
    backend_name = str(model_cfg.get("backend", "local"))
    max_model_len = int(model_cfg.max_model_len)
    max_tokens = int(cfg.sampling_params.max_tokens)
    max_prompt_tokens = max_model_len - max_tokens
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"sampling_params.max_tokens ({max_tokens}) must be < model.max_model_len ({max_model_len})."
        )

    if backend_name == "local":
        from vllm import LLM, SamplingParams

        # MedGemma/processor import can init CUDA in the parent before vLLM forks
        # EngineCore; fork then fails with "Cannot re-initialize CUDA in forked subprocess".
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        sampling_params = SamplingParams(**OmegaConf.to_container(cfg.sampling_params, resolve=True))
        llm_kwargs = OmegaConf.to_container(model_cfg, resolve=True)
        llm_kwargs.pop("backend", None)
        log.info(f"Loading local model ({llm_kwargs.get('model')})...")
        log.info(sampling_params)
        backend: GenerationBackend = LocalBackend(LLM(**llm_kwargs))
    elif backend_name == "arc":
        sampling_params = cfg.sampling_params
        model_name = str(model_cfg.model)
        api_base = str(model_cfg.api_base)
        api_key_env = str(model_cfg.get("api_key_env", "ARC_AI_API_KEY"))
        api_key = str(model_cfg.get("api_key", "") or os.environ[api_key_env])
        tokenizer_cfg = model_cfg.get("tokenizer", model_cfg.get("tokenizer_name", None))
        tokenizer_name = None if tokenizer_cfg in (None, "", "null") else str(tokenizer_cfg)
        log.info(
            "Using ARC backend model=%s api_base=%s tokenizer=%s",
            model_name,
            api_base,
            tokenizer_name or "none (no local truncation)",
        )
        log.info(sampling_params)
        backend = ArcBackend(
            model=model_name,
            api_base=api_base,
            api_key=api_key,
            tokenizer_name=tokenizer_name,
            max_concurrency=int(model_cfg.get("max_concurrency", 8)),
            timeout_s=float(model_cfg.get("timeout_s", 120.0)),
            max_retries=int(model_cfg.get("max_retries", 5)),
            request_interval_s=float(model_cfg.get("request_interval_s", 0.0)),
        )
    else:
        raise ValueError(
            f"Unknown model.backend {backend_name!r}. Expected 'local' (default) or 'arc'."
        )

    log.info(f"Truncating prompts to {max_prompt_tokens} tokens when needed (keeping the end).")
    return backend, sampling_params, max_prompt_tokens
