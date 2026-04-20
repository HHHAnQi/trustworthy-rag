"""
src/llm.py  —  LLM wrapper (Qwen2.5 / any HuggingFace causal-LM)

WHY we loop K times instead of num_return_sequences=K
------------------------------------------------------
Conformal prediction requires that calibration and inference scores are
exchangeable, which in turn requires that the K LLM samples are i.i.d.
draws from p(y | prompt).

HuggingFace's num_return_sequences generates K sequences in a *single*
batched forward pass with a shared KV-cache prefix.  The samples share
internal state and are not strictly independent.  Looping K separate
model.generate() calls gives true independent draws.

Runtime cost: K× latency per query (acceptable for K=5 on one GPU).
If speed matters, switch to vLLM which handles independent sampling correctly.
"""

from __future__ import annotations
import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, config: dict):
        """
        config keys
        -----------
        name            str    HF model id or local path
        max_new_tokens  int    (default 256)
        temperature     float  (default 0.7)
        num_samples     int    default K  (default 5)
        """
        model_name: str = config["name"]
        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model: %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.max_new_tokens: int      = config.get("max_new_tokens", 256)
        self.temperature: float       = config.get("temperature", 0.7)
        self.default_num_samples: int = config.get("num_samples", 5)

    def generate_samples(self, prompt: str, num_samples: int | None = None) -> List[str]:
        """
        Return K independently sampled answer strings (prompt stripped).
        Each call to model.generate() is a separate forward pass → i.i.d. samples.
        """
        k = num_samples if num_samples is not None else self.default_num_samples

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        prompt_len: int = inputs["input_ids"].shape[1]

        results: List[str] = []
        with torch.no_grad():
            for _ in range(k):
                out = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                # out shape: (1, prompt_len + generated_len)
                text = self.tokenizer.decode(
                    out[0, prompt_len:], skip_special_tokens=True
                ).strip()
                results.append(text)
        return results
