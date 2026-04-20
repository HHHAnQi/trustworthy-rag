"""
src/rag.py  —  FAISS dense retriever

Index type: IndexFlatIP (inner product on L2-normalised embeddings = cosine similarity).
Higher score → more similar.  build_index.py must use normalize_embeddings=True.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RAG:
    def __init__(self, config: dict):
        """
        config keys
        -----------
        index_path  str  path to .index file (built by scripts/build_index.py)
        docs_path   str  plain-text corpus, one document per line
        embedder    str  sentence-transformers model id
        top_k       int  number of documents to retrieve  (default 3)
        """
        embedder_name: str = config.get("embedder", "BAAI/bge-small-en-v1.5")
        logger.info("RAG: loading embedder %s", embedder_name)
        self.embedder = SentenceTransformer(embedder_name)

        index_path = Path(config["index_path"])
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run `python scripts/build_index.py` first."
            )
        logger.info("RAG: loading index from %s", index_path)
        self.index = faiss.read_index(str(index_path))

        docs_path = Path(config["docs_path"])
        self.docs: List[str] = self._load_docs(docs_path)
        logger.info("RAG: loaded %d documents", len(self.docs))

        self.top_k: int = config.get("top_k", 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> List[str]:
        """Return the top-k most relevant document strings."""
        texts, _ = self._search(query, top_k)
        return texts

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> Tuple[List[str], List[float]]:
        """Return (texts, scores) — higher score = more similar (IndexFlatIP)."""
        return self._search(query, top_k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _search(self, query: str, top_k: int | None) -> Tuple[List[str], List[float]]:
        k = top_k if top_k is not None else self.top_k
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(q_emb, dtype="float32"), k)
        texts = [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]
        return texts, scores[0].tolist()

    @staticmethod
    def _load_docs(path: Path) -> List[str]:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]
