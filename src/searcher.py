"""
src/searcher.py
---------------
扩展检索器，实现两阶段自适应检索策略的第二阶段。

设计思路：
- 第一阶段：RAG top-3 精确检索，CP 评估置信度
- 若 CP abstained，触发第二阶段：top-10 扩展检索
- 更多文档 → 更丰富的 context → 更高的回答成功率

通过 MCP Tool（extended_search）对外暴露接口，
架构上支持未来替换为外部搜索服务（SerpAPI 等）。
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)


class WebSearcher:
    """
    本地扩展检索器。
    接口设计与 Web Search 兼容，未来可无缝替换为外网搜索。
    """

    def __init__(self, max_results: int = 10, rag=None):
        self.max_results = max_results
        self.rag = rag

    def search(self, query: str) -> List[str]:
        """从本地知识库扩展检索，返回 top-k 文档列表。"""
        if self.rag is None:
            logger.warning("RAG not injected, returning empty results")
            return []
        try:
            docs = self.rag.retrieve(query, top_k=self.max_results)
            logger.info("Extended retrieval %r → %d docs", query, len(docs))
            return docs
        except Exception as e:
            logger.warning("Extended retrieval failed: %s", e)
            return []
