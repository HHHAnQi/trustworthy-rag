"""
src/router.py
-------------
有限状态路由器，基于规则决策，不调用 LLM。

前置路由（pre_route）：
  safe_refuse  — Guardian 已判定不安全
  clarify      — 问题过于模糊，无法检索
  proceed      — 正常进入 pipeline

后置路由（post_route）：
  answer        — CP reliable，直接返回
  retrieve_more — CP abstained，检索支持度低，值得重试
  abstain       — CP abstained，已重试或支持度极低，放弃
"""
from __future__ import annotations
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 模糊输入的关键词，单独出现时触发 clarify
_VAGUE_PATTERNS = [
    r"^(为什么|怎么办|帮我|这个|那个|如何|什么|是什么|怎么)$",
    r"^(why|how|what|help|this|that|tell me)[\s\?\.\!]*$",
]

# 最少实质性词数（去掉标点后）
_MIN_CONTENT_WORDS = 2


def pre_route(query: str, guardian_result: Dict[str, Any]) -> str:
    """
    前置路由：在调用 pipeline 之前决定处理方式。

    Parameters
    ----------
    query           : 用户输入
    guardian_result : {"is_safe": bool, "reason": str}

    Returns
    -------
    "safe_refuse" | "clarify" | "proceed"
    """
    # 1. Guardian 已判定不安全
    if not guardian_result.get("is_safe", True):
        logger.info("Pre-route: safe_refuse | reason=%s",
                    guardian_result.get("reason"))
        return "safe_refuse"

    # 2. 空输入
    stripped = query.strip()
    if not stripped:
        logger.info("Pre-route: clarify | empty input")
        return "clarify"

    # 3. 去掉标点后的实质性词数太少
    words = re.findall(r"[\w\u4e00-\u9fff]+", stripped)
    if len(words) < _MIN_CONTENT_WORDS:
        logger.info("Pre-route: clarify | too few words (%d)", len(words))
        return "clarify"

    # 4. 只有模糊词，没有具体对象
    for pattern in _VAGUE_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            logger.info("Pre-route: clarify | vague input: %r", stripped)
            return "clarify"

    logger.debug("Pre-route: proceed | query=%r", query[:50])
    return "proceed"


def post_route(result: Dict[str, Any], retry_count: int) -> str:
    """
    后置路由：根据 pipeline 结果决定下一步。

    综合考虑：
      - CP status（reliable / abstained）
      - nonconformity score vs threshold
      - retrieval_support（置信度的检索分量）
      - retry_count（已重试次数）

    Returns
    -------
    "answer" | "retrieve_more" | "abstain"
    """
    status  = result.get("status", "abstained")
    score   = result.get("score", 1.0)
    threshold = result.get("threshold", 0.5)

    # CP 通过，有答案
    if status == "reliable" and result.get("answer"):
        logger.debug("Post-route: answer | score=%.4f threshold=%.4f",
                     score, threshold)
        return "answer"

    # CP 没过，判断是否值得重试
    if retry_count >= 1:
        # 已经重试过一次，不再重试
        logger.info("Post-route: abstain | retry_count=%d", retry_count)
        return "abstain"

    # 检索支持度分量
    breakdown = result.get("breakdown") or {}
    retrieval_support = breakdown.get("retrieval_support", 0.5)

    # score 接近 threshold（差距小于 0.15），说明差一点就过了，值得重试
    score_gap = score - threshold
    worth_retry = score_gap < 0.15 and retrieval_support > 0.2

    if worth_retry:
        logger.info(
            "Post-route: retrieve_more | score_gap=%.4f retrieval_support=%.4f",
            score_gap, retrieval_support
        )
        return "retrieve_more"

    # score 远超 threshold，或者检索支持度极低，重试没有意义
    logger.info(
        "Post-route: abstain | score_gap=%.4f retrieval_support=%.4f",
        score_gap, retrieval_support
    )
    return "abstain"
