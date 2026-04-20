"""
scripts/calibrate.py  —  Offline conformal calibration

CRITICAL REQUIREMENT: exchangeability
--------------------------------------
Split conformal prediction guarantees P(error) ≤ α only when the
nonconformity scores from calibration and inference are exchangeable,
i.e. drawn from the same distribution.

This means the calibration pipeline must be IDENTICAL to inference:
  1. Retrieve docs for the question via RAG
  2. Build the same RAG-augmented prompt  ← OFTEN MISSED
  3. Sample K answers from the LLM
  4. Compute uncertainty(answers, docs)
  5. s = 1 - confidence

If you skip step 1-2 and sample from the raw question, the score
distribution differs and the guarantee becomes invalid.

Usage
-----
    # Full pipeline (recommended)
    python scripts/calibrate.py \
        --config    configs/config.yaml \
        --data_path data/calibration.json \
        --output    data/qhat.json \
        --alphas    0.05 0.1 0.2

    # Consistency-only mode (when RAG corpus not yet ready):
    # Set weights.retrieval=0 in config.yaml, then run the same command.
    # The score distributions will still be consistent between cal and inference.

Input format (data/calibration.json)
-------------------------------------
    [
        {"question": "Who wrote Hamlet?",  "answer": "Shakespeare"},
        ...
    ]
    The "answer" field is only used for optional evaluation, not for calibration.

Output (data/qhat.json)
------------------------
    {"0.05": 0.312, "0.1": 0.418, "0.2": 0.531}
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conformal import compute_qhat
from src.llm import LLM
from src.rag import RAG
from src.uncertainty import Uncertainty
from src.utils import load_config, setup_logging

# Reuse the same prompt template as pipeline.py
_PROMPT_TEMPLATE = (
    "Answer the question in ONE sentence using only the context below. "
    "Do not explain your reasoning. Do not mention the context. "
    "If the answer is not in the context, say: I don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (one sentence):"
)

setup_logging()
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d calibration samples from %s", len(data), path)
    return data


def main():
    parser = argparse.ArgumentParser(description="Conformal calibration")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--data_path", required=True, help="calibration JSON path")
    parser.add_argument("--output",    default="data/qhat.json")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    config = load_config(args.config)
    num_samples: int = config["llm"].get("num_samples", 5)

    # Initialise all modules (same config as inference)
    llm         = LLM(config["llm"])
    rag         = RAG(config["rag"])
    uncertainty = Uncertainty(config["uncertainty"])

    dataset = load_dataset(args.data_path)
    scores: List[float] = []

    logger.info("Running calibration pipeline on %d samples …", len(dataset))

    for item in tqdm(dataset, desc="Calibrating"):
        question: str = item["question"]
        try:
            # ── STEP 1: RAG retrieval ─────────────────────────────────
            docs = rag.retrieve(question)

            # ── STEP 2: Build the SAME RAG-augmented prompt as inference
            #    This is the critical step that ensures exchangeability.
            context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
            prompt  = _PROMPT_TEMPLATE.format(context=context, question=question)

            # ── STEP 3: K independent LLM samples (same as inference) ─
            samples = llm.generate_samples(prompt, num_samples=num_samples)

            # ── STEP 4: Uncertainty (same function as inference) ───────
            score = uncertainty.compute_nonconformity(samples, docs)
            scores.append(score)

        except Exception as exc:
            logger.warning("Skipped %r — %s", question[:60], exc)

    if not scores:
        raise RuntimeError("No calibration scores collected. Check config and data.")

    arr = np.array(scores)
    logger.info(
        "Calibration scores — n=%d  min=%.4f  mean=%.4f  max=%.4f",
        len(arr), arr.min(), arr.mean(), arr.max(),
    )

    # ── Compute q̂ per alpha (method='higher' for conservative coverage) ──
    qhat_table: dict = {}
    for alpha in args.alphas:
        qhat = compute_qhat(arr.tolist(), alpha)
        qhat_table[str(alpha)] = round(qhat, 6)
        logger.info("α=%.2f  →  q̂=%.4f", alpha, qhat)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(qhat_table, f, indent=2)
    logger.info("Saved q̂ table to %s", out_path)


if __name__ == "__main__":
    main()
