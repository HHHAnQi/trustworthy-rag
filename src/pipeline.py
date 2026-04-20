"""
src/pipeline.py  —  End-to-end inference pipeline

Flow
----
query → RAG retrieve → build prompt → LLM K-sample → uncertainty → conformal → result
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional

from src.conformal import ConformalPredictor
from src.llm import LLM
from src.rag import RAG
from src.uncertainty import Uncertainty, FaithfulnessChecker

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "Answer the question in ONE sentence using only the context below. "
    "Do not explain your reasoning. Do not mention the context. "
    "If the answer is not in the context, say: I don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (one sentence):"
)


class TrustworthyRAGPipeline:
    def __init__(self, config: Dict):
        self.llm        = LLM(config["llm"])
        self.rag        = RAG(config["rag"])
        self.uncertainty = Uncertainty(config["uncertainty"])
        self.conformal  = ConformalPredictor(config["conformal"])
        self.num_samples: int = config["llm"].get("num_samples", 5)

        nli_path = config.get("faithfulness", {}).get("model_path")
        if nli_path:
            self.faithfulness = FaithfulnessChecker(nli_path)
            logger.info("FaithfulnessChecker loaded.")
        else:
            self.faithfulness = None

        logger.info("TrustworthyRAGPipeline ready.")

    def run(
        self,
        query: str,
        alpha: Optional[float] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline for one query.

        Returns
        -------
        dict with keys:
          query, answer (None if abstained), confidence, accepted,
          status ("reliable"|"abstained"), threshold, alpha, score,
          + docs/samples/breakdown if return_details=True
        """
        # 1. RAG retrieval
        docs = self.rag.retrieve(query)

        # 2. Build RAG-augmented prompt
        context = "\n\n".join(f"[{i+1}] {d}" for i, d in enumerate(docs))
        prompt  = _PROMPT_TEMPLATE.format(context=context, question=query)

        # 3. K independent samples
        samples = self.llm.generate_samples(prompt, num_samples=self.num_samples)

        # 4. Uncertainty  (same prompt structure as calibration — exchangeability preserved)
        confidence   = self.uncertainty.compute_confidence(samples, docs)
        nonconformity = 1.0 - confidence

        # 5. Conformal decision
        decision = self.conformal.predict(nonconformity, alpha=alpha)

        # 6. Answer selection: most representative sample (soft majority vote)
        best_answer = self._select_answer(samples)
        best_answer = self._postprocess(best_answer)

        # 空答案强制 abstain
        final_accepted = decision["accepted"] and bool(best_answer)

        # Faithfulness 检测
        faithfulness = None
        if final_accepted and self.faithfulness and best_answer:
            faithfulness = self.faithfulness.check(best_answer, docs)

        result: Dict[str, Any] = {
            "query":        query,
            "answer":       best_answer if final_accepted else None,
            "confidence":   round(confidence, 4),
            "accepted":     final_accepted,
            "status":       "reliable" if final_accepted else "abstained",
            "threshold":    decision["threshold"],
            "alpha":        decision["alpha"],
            "score":        decision["score"],
            "faithfulness": faithfulness,
        }
        if return_details:
            result["docs"]      = docs
            result["samples"]   = samples
            result["breakdown"] = self.uncertainty.breakdown(samples, docs)
        return result

    def _postprocess(self, answer: str) -> str:
        """
        清理 LLM 生成的答案：
        去掉末尾无用的拼接（如 "I don't know"），保留核心内容。
        不强制截断到第一句，避免把答案截空。
        """
        if not answer:
            return answer

        # 去掉常见的无用后缀拼接
        cutoffs = [
            "I don't know",
            "However, the context",
            "Note that",
            "Please note",
            "It is worth noting",
            "It should be noted",
            "The context does not",
            "Based on the context",
            "According to the context",
            "The provided context",
        ]
        for cutoff in cutoffs:
            if cutoff in answer:
                answer = answer[:answer.index(cutoff)].strip()

        # 去掉末尾多余的标点
        answer = answer.strip().rstrip(',').strip()

        return answer if answer else answer

    def _select_answer(self, samples: List[str]) -> str:
        """Return the sample with highest average similarity to all others."""
        if len(samples) == 1:
            return samples[0]
        from sklearn.metrics.pairwise import cosine_similarity
        embs    = self.uncertainty.embedder.encode(samples, normalize_embeddings=True)
        sim_mat = cosine_similarity(embs)
        avg_sim = (sim_mat.sum(axis=1) - 1.0) / (len(samples) - 1)
        return samples[int(avg_sim.argmax())]
