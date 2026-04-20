# Trustworthy RAG

A RAG-based QA system with uncertainty estimation and conformal prediction.

## Features
- RAG retrieval (FAISS)
- LLM multi-sampling (Qwen)
- Uncertainty estimation (consistency + retrieval support)
- Conformal prediction (risk control)

## Run

### Build index
python scripts/build_index.py

### Start API
uvicorn src.api.main:app --reload