"""
src/rewriter.py
---------------
当 CP 判定 abstained 时，用 LangChain 改写问题。
改写策略：把问题简化为更短的核心实体查询。
不依赖外部 API，直接调用本地 Qwen。
"""
from __future__ import annotations
import logging
from typing import List

logger = logging.getLogger(__name__)

# 改写 prompt
_REWRITE_PROMPT = (
    "Rewrite the following question into a shorter, simpler factual query "
    "that focuses on the key entity or fact being asked about. "
    "Return ONLY the rewritten question, nothing else.\n\n"
    "Original: {question}\n"
    "Rewritten:"
)

class QueryRewriter:
    """用本地 LLM 改写查询，LangChain PromptTemplate 风格。"""

    def __init__(self, llm):
        """
        llm: src.llm.LLM 实例，复用已加载的模型，不重复加载。
        """
        self.llm = llm

    def rewrite(self, question: str) -> str:
        """
        把原始问题改写为更简洁的查询。
        返回改写后的问题字符串。
        """
        prompt = _REWRITE_PROMPT.format(question=question)
        samples = self.llm.generate_samples(prompt, num_samples=1)
        rewritten = samples[0].strip() if samples else question

        # 去掉可能的引号包裹
        rewritten = rewritten.strip('"\'')

        # 如果改写后反而更长，退回原始问题
        if len(rewritten) > len(question) * 1.5:
            logger.debug("改写后更长，退回原始: %r", question)
            return question

        logger.info("Query rewrite: %r → %r", question, rewritten)
        return rewritten if rewritten else question
