"""
src/mcp_server.py
-----------------
独立 MCP Tool Server，通过 HTTP 调用主 API（端口 6006），
不重复加载模型，解决显存不足问题。

启动方式（主 API 启动后再启动）：
    python src/mcp_server.py

监听端口：8001
依赖：主 API 运行在 http://localhost:6006
"""
from __future__ import annotations
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastmcp import FastMCP
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

mcp      = FastMCP("Trustworthy RAG Tools")
API_BASE = "http://localhost:6006"


def _call_api(endpoint: str, payload: dict) -> dict:
    """向主 API 发送 POST 请求。"""
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


@mcp.tool()
def extended_search(query: str) -> List[str]:
    """
    Search for additional context using extended local retrieval.

    When the initial RAG retrieval (top-3) is insufficient for
    Conformal Prediction to accept an answer, this tool retrieves
    top-10 documents from the local knowledge base to provide
    broader context for a second attempt.

    Args:
        query: The search query string.

    Returns:
        List of up to 10 document snippets from the knowledge base.
        Each snippet is at most 400 characters.
    """
    try:
        result = _call_api("/ask", {
            "query":          query,
            "alpha":          0.2,
            "return_details": True,
        })
        docs = result.get("docs", [])
        logger.info("extended_search %r → %d docs", query, len(docs))
        return docs if docs else []
    except Exception as e:
        logger.warning("extended_search failed: %s", e)
        return []


@mcp.tool()
def trustworthy_qa(question: str, alpha: float = 0.1) -> dict:
    """
    Answer a question using RAG with conformal prediction risk control.

    Args:
        question: The question to answer.
        alpha: Miscoverage rate (0.05/0.1/0.2).
               Lower alpha = stricter abstention.

    Returns:
        answer: Answer string or null if abstained.
        status: 'reliable' or 'abstained'.
        confidence: Confidence score in [0, 1].
        score: Nonconformity score.
        threshold: Conformal threshold q_hat.
        docs: Retrieved document snippets.
    """
    try:
        result = _call_api("/ask", {
            "query":          question,
            "alpha":          alpha,
            "return_details": True,
        })
        return {
            "answer":     result.get("answer"),
            "status":     result.get("status"),
            "confidence": result.get("confidence"),
            "score":      result.get("score"),
            "threshold":  result.get("threshold"),
            "docs":       result.get("docs", []),
        }
    except Exception as e:
        logger.warning("trustworthy_qa failed: %s", e)
        return {"answer": None, "status": "error", "confidence": 0.0,
                "score": 1.0, "threshold": 0.0, "docs": []}


if __name__ == "__main__":
    logger.info("Starting MCP Tool Server on http://localhost:8001")
    logger.info("Proxying requests to main API at %s", API_BASE)
    mcp.run(transport="http", host="0.0.0.0", port=8001)
