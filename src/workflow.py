"""
src/workflow.py
---------------
Agentic RAG 编排层。

职责：
  Guardian → pre_route → pipeline → post_route
  → retrieve_more（最多1次）→ logger

流程：
  1. Guardian 安全检测
  2. 前置路由（clarify / safe_refuse / proceed）
  3. 第一次调用 pipeline
  4. 后置路由（answer / retrieve_more / abstain）
  5. 如果 retrieve_more：扩展检索后重新调用 pipeline
  6. 全链路 logger 落盘
  7. 返回结构化结果

pipeline.py 完全不改，只在这里做编排。
evaluate.py 和 api/main.py 都调用这一层，
保证离线评测和在线服务走同一套逻辑。
"""
from __future__ import annotations
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# 路由决策对应的提示语
_CLARIFY_MSG = (
    "Your question is too vague. "
    "Could you provide more details or specify what you are looking for?"
)
_SAFE_REFUSE_MSG = (
    "I cannot process this request. "
    "It appears to contain potentially unsafe content."
)


class AgenticWorkflow:
    """
    Agentic RAG 编排器。
    把 Guardian、Router、Pipeline、Logger 组合成完整流程。
    """

    def __init__(self, pipeline, guardian, router, trace_logger):
        """
        pipeline     : TrustworthyRAGPipeline
        guardian     : Guardian
        router       : 模块级函数（pre_route / post_route）
        trace_logger : TraceLogger
        """
        self.pipeline     = pipeline
        self.guardian     = guardian
        self.trace_logger = trace_logger

        # router 是模块，直接导入函数
        from src.router import pre_route, post_route
        self._pre_route  = pre_route
        self._post_route = post_route

    def run(
        self,
        query:      str,
        alpha:      float = 0.1,
        session_id: str   = "default",
    ) -> Dict[str, Any]:
        """
        完整的 Agentic RAG 流程。

        Returns
        -------
        dict，包含：
          answer, status, confidence, score, threshold,
          faithfulness, route, retry_count, latency_ms,
          docs, guardian
        """
        t0 = time.perf_counter()

        # ── Step 1: Guardian 安全检测 ─────────────────────────────────
        is_safe, reason = self.guardian.check(query)
        guardian_result = {"is_safe": is_safe, "reason": reason}

        # ── Step 2: 前置路由 ──────────────────────────────────────────
        pre = self._pre_route(query, guardian_result)

        if pre == "safe_refuse":
            return self._make_result(
                query      = query,
                answer     = None,
                status     = "safe_refuse",
                route      = "safe_refuse",
                message    = _SAFE_REFUSE_MSG,
                guardian   = guardian_result,
                latency_ms = self._ms(t0),
                session_id = session_id,
                alpha      = alpha,
            )

        if pre == "clarify":
            return self._make_result(
                query      = query,
                answer     = None,
                status     = "clarify",
                route      = "clarify",
                message    = _CLARIFY_MSG,
                guardian   = guardian_result,
                latency_ms = self._ms(t0),
                session_id = session_id,
                alpha      = alpha,
            )

        # ── Step 3: 第一次调用 pipeline ───────────────────────────────
        result = self.pipeline.run(
            query=query, alpha=alpha, return_details=True
        )
        retry_count = 0

        # ── Step 4: 后置路由 ──────────────────────────────────────────
        post = self._post_route(result, retry_count=retry_count)

        # ── Step 5: retrieve_more（最多重试一次）──────────────────────
        if post == "retrieve_more":
            logger.info("Workflow: retrieve_more triggered, retrying...")
            result = self._run_with_extended(query, alpha)
            retry_count = 1
            post = self._post_route(result, retry_count=retry_count)

        # post_route 最终只有 answer 或 abstain
        if post != "answer":
            result["answer"] = None
            result["status"] = "abstained"

        latency_ms = self._ms(t0)

        # ── Step 6: 全链路 logger ─────────────────────────────────────
        # JSONL 本地日志
        self.trace_logger.log(
            session_id       = session_id,
            question         = query,
            guardian         = guardian_result,
            pre_route        = pre,
            post_route       = post,
            pipeline_result  = result,
            retry_count      = retry_count,
            latency_ms       = latency_ms,
        )
        # Langfuse 云端追踪
        if hasattr(self, 'langfuse_tracer') and self.langfuse_tracer:
            self.langfuse_tracer.trace_workflow(
                session_id       = session_id,
                question         = query,
                guardian         = guardian_result,
                pre_route        = pre,
                post_route       = post,
                pipeline_result  = result,
                retry_count      = retry_count,
                latency_ms       = latency_ms,
            )

        # ── Step 7: 返回结果 ──────────────────────────────────────────
        faith = result.get("faithfulness") or {}
        return {
            "query":       query,
            "answer":      result.get("answer"),
            "status":      result.get("status", "abstained"),
            "confidence":  result.get("confidence", 0.0),
            "score":       result.get("score", 1.0),
            "threshold":   result.get("threshold", 0.5),
            "alpha":       alpha,
            "route":       post,
            "retry_count": retry_count,
            "latency_ms":  latency_ms,
            "docs":        result.get("docs", []),
            "faithfulness": {
                "label": faith.get("label"),
                "score": faith.get("faithfulness_score"),
            } if faith else None,
            "guardian":    guardian_result,
        }

    def _run_with_extended(
        self, query: str, alpha: float
    ) -> Dict[str, Any]:
        """扩展检索后重新调用 pipeline（top-10）。"""
        try:
            extra_docs = self.pipeline.rag.retrieve(query, top_k=10)
            return self.pipeline._run_with_extra_docs(
                query, alpha, extra_docs
            )
        except Exception as e:
            logger.warning("Extended retrieval failed: %s", e)
            return self.pipeline.run(
                query=query, alpha=alpha, return_details=True
            )

    def _make_result(
        self,
        query:      str,
        answer:     Optional[str],
        status:     str,
        route:      str,
        session_id: str,
        alpha:      float,
        latency_ms: float,
        message:    str = "",
        guardian:   Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """构造早返回结果（clarify / safe_refuse）。"""
        self.trace_logger.log(
            session_id      = session_id,
            question        = query,
            guardian        = guardian or {},
            pre_route       = route,
            post_route      = route,
            pipeline_result = {},
            retry_count     = 0,
            latency_ms      = latency_ms,
        )
        return {
            "query":       query,
            "answer":      answer,
            "status":      status,
            "confidence":  0.0,
            "score":       1.0,
            "threshold":   0.0,
            "alpha":       alpha,
            "route":       route,
            "retry_count": 0,
            "latency_ms":  latency_ms,
            "docs":        [],
            "faithfulness": None,
            "guardian":    guardian or {},
            "message":     message,
        }

    @staticmethod
    def _ms(t0: float) -> float:
        return round((time.perf_counter() - t0) * 1000, 1)
