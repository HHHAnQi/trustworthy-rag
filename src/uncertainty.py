"""
src/uncertainty.py  —  Uncertainty / confidence estimation

Two signals
-----------
1. Answer consistency  (self-consistency, ConU §3.1)
   Pairwise semantic similarity across K independently sampled answers.
   Diagonal masked to avoid self-similarity inflation.

2. Retrieval support  (grounding check)
   How well is the answer semantically supported by retrieved documents?

Both signals are normalised to [0, 1] before weighting:
  - consistency  : already in [0,1] (cosine sim of normalised embeddings)
  - retrieval_support: raw cosine sim clipped to [0,1]
    (bge embeddings are normalised, so inner product ∈ [-1,1]; we clip to 0)

Final score
-----------
  confidence         = w1 * consistency + w2 * retrieval_support  ∈ [0, 1]
  nonconformity (s)  = 1 - confidence                              ∈ [0, 1]

The nonconformity score is fed into conformal calibration / prediction.

NOTE ON CALIBRATION CONSISTENCY
--------------------------------
During calibrate.py the uncertainty is computed on the same
(RAG context, K-sample answers) pairs as inference.  If you disable RAG
at calibration time (e.g. pass docs=[]) you MUST also set w2=0 in config,
otherwise the score distributions diverge and the conformal guarantee fails.
"""

from __future__ import annotations
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class Uncertainty:
    def __init__(self, config: dict):
        """
        config keys
        -----------
        embedder           str    sentence-transformers model id
        weights.consistency  float  w1  (default 0.5)
        weights.retrieval    float  w2  (default 0.5)
        """
        embedder_name: str = config.get("embedder", "BAAI/bge-small-en-v1.5")
        logger.info("Uncertainty: loading embedder %s", embedder_name)
        self.embedder = SentenceTransformer(embedder_name)

        weights = config.get("weights", {})
        self.w1: float = weights.get("consistency", 0.5)
        self.w2: float = weights.get("retrieval", 0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_confidence(self, answers: List[str], docs: List[str]) -> float:
        """
        Returns confidence ∈ [0, 1].

        answers : K sampled LLM outputs for the same RAG-augmented prompt
        docs    : retrieved documents (same ones used to build the prompt)
        """
        if not answers:
            raise ValueError("answers must be non-empty")
        c = self._consistency(answers)
        r = self._retrieval_support(answers[0], docs)
        return float(np.clip(self.w1 * c + self.w2 * r, 0.0, 1.0))

    def compute_nonconformity(self, answers: List[str], docs: List[str]) -> float:
        """Nonconformity score s = 1 - confidence ∈ [0, 1]."""
        return 1.0 - self.compute_confidence(answers, docs)

    def breakdown(self, answers: List[str], docs: List[str]) -> dict:
        """Diagnostic: return all component scores."""
        c = self._consistency(answers)
        r = self._retrieval_support(answers[0], docs)
        conf = float(np.clip(self.w1 * c + self.w2 * r, 0.0, 1.0))
        return {
            "consistency":       round(c, 4),
            "retrieval_support": round(r, 4),
            "confidence":        round(conf, 4),
            "nonconformity":     round(1.0 - conf, 4),
            "w1": self.w1,
            "w2": self.w2,
        }

    # ------------------------------------------------------------------
    # Private: component signals
    # ------------------------------------------------------------------

    def _consistency(self, answers: List[str]) -> float:
        """
        Mean pairwise cosine similarity across K answers (diagonal excluded).
        Returns 0.8 for K=1 (no variance information, slight discount).
        """
        if len(answers) == 1:
            return 0.8
        embs = self.embedder.encode(answers, normalize_embeddings=True)
        sim = cosine_similarity(embs)               # K×K, diag=1
        k = len(answers)
        mask = ~np.eye(k, dtype=bool)               # off-diagonal mask
        return float(np.mean(sim[mask]))             # already in [0,1]

    def _retrieval_support(self, answer: str, docs: List[str]) -> float:
        """
        Mean cosine similarity between answer and each retrieved doc.
        Clipped to [0, 1] — bge embeddings are L2-normalised, so raw
        inner product ∈ [-1, 1]; negative values mean no support.

        Returns 0.0 when docs is empty (e.g. w2=0 mode).
        """
        if not docs:
            return 0.0
        emb_a = self.embedder.encode([answer], normalize_embeddings=True)
        emb_d = self.embedder.encode(docs,     normalize_embeddings=True)
        sims  = cosine_similarity(emb_a, emb_d)   # shape (1, n_docs)
        return float(np.clip(np.mean(sims), 0.0, 1.0))


class FaithfulnessChecker:
    """
    用 NLI 模型检测答案是否忠实于检索文档。

    衡量标准：
      检索文档是否"蕴含"答案（Entailment）
      三类标签：Entailment / Neutral / Contradiction

    Faithfulness score = Entailment 概率
    越高说明答案越忠实于检索文档，幻觉可能性越低。

    设计意图：
      与 CP 框架结合：faithfulness 低时主动警告用户
      答案可能不完全基于检索内容（存在幻觉风险）
    """

    def __init__(self, model_path: str):
        from sentence_transformers import CrossEncoder
        logger.info("FaithfulnessChecker: loading NLI model from %s", model_path)
        self.model = CrossEncoder(
            model_path,
            num_labels=3,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
        # DeBERTa NLI 标签顺序：0=Contradiction, 1=Neutral, 2=Entailment
        self.entailment_idx = 2

    def check(self, answer: str, docs: list) -> dict:
        """
        检测答案是否忠实于检索文档。

        Parameters
        ----------
        answer : 模型生成的答案
        docs   : 检索到的文档列表

        Returns
        -------
        dict 含：
          faithfulness_score  float  ∈ [0,1]，越高越忠实
          label               str    "faithful" | "uncertain" | "hallucinated"
          max_doc_score       float  与最相关文档的 entailment 概率
          detail              list   每条文档的得分
        """
        if not answer or not docs:
            return {
                "faithfulness_score": 0.0,
                "label": "uncertain",
                "max_doc_score": 0.0,
                "detail": [],
            }

        import numpy as np
        from scipy.special import softmax

        scores = []
        for doc in docs[:3]:   # 只用 top-3，节省推理时间
            # premise = 文档，hypothesis = 答案
            logits = self.model.predict([(doc, answer)], apply_softmax=False)[0]
            probs  = softmax(logits)
            entailment_prob = float(probs[self.entailment_idx])
            scores.append(entailment_prob)

        max_score = max(scores)
        avg_score = float(np.mean(scores))

        # 用最高分（只需一条文档支持答案即可）
        faithfulness_score = max_score

        if faithfulness_score >= 0.6:
            label = "faithful"
        elif faithfulness_score >= 0.3:
            label = "uncertain"
        else:
            label = "hallucinated"

        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "label":              label,
            "max_doc_score":      round(max_score, 4),
            "detail":             [round(s, 4) for s in scores],
        }
