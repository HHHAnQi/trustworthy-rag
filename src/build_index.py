"""
scripts/build_index.py  —  Build FAISS index from docs.txt

Usage
-----
    python scripts/build_index.py \
        --docs_path  data/docs.txt \
        --index_path data/faiss.index \
        --embedder   BAAI/bge-small-en-v1.5

One document per line in docs.txt.  Blank lines are ignored.
Uses IndexFlatIP with L2-normalised embeddings (= cosine similarity).
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build(docs_path: str, index_path: str, embedder_name: str, batch_size: int):
    docs_path  = Path(docs_path)
    index_path = Path(index_path)

    if not docs_path.exists():
        raise FileNotFoundError(f"Corpus not found: {docs_path}")

    docs = [ln.strip() for ln in docs_path.read_text(encoding="utf-8").splitlines()]
    docs = [d for d in docs if d]
    logger.info("Loaded %d documents", len(docs))

    logger.info("Loading embedder: %s", embedder_name)
    embedder = SentenceTransformer(embedder_name)

    logger.info("Encoding (batch_size=%d)…", batch_size)
    embs = embedder.encode(
        docs, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )
    embs = np.array(embs, dtype="float32")
    logger.info("Embeddings shape: %s", embs.shape)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on normalised vecs = cosine
    index.add(embs)
    logger.info("FAISS index: %d vectors, dim=%d", index.ntotal, dim)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("Index saved → %s", index_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_path",  default="data/docs.txt")
    p.add_argument("--index_path", default="data/faiss.index")
    p.add_argument("--embedder",   default="BAAI/bge-small-en-v1.5")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    build(args.docs_path, args.index_path, args.embedder, args.batch_size)


if __name__ == "__main__":
    main()
