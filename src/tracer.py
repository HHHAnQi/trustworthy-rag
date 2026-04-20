"""
src/tracer.py — Langfuse v4
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LangfuseTracer:
    def __init__(self):
        self.enabled = False
        self.client  = None
        self._load_env()

        sk = os.getenv("LANGFUSE_SECRET_KEY")
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

        if sk and pk:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(secret_key=sk, public_key=pk, host=host)
                self.client.auth_check()
                self.enabled = True
                logger.info("✅ Langfuse v4 已启用 host=%s", host)
            except Exception as e:
                logger.warning("Langfuse 初始化失败: %s", e)

    def _load_env(self):
        env_path = "configs/.env"
        if not os.path.exists(env_path):
            return
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    def trace_workflow(
        self,
        session_id:      str,
        question:        str,
        guardian:        Dict[str, Any],
        pre_route:       str,
        post_route:      str,
        pipeline_result: Dict[str, Any],
        retry_count:     int,
        latency_ms:      float,
    ) -> None:
        if not self.enabled or not self.client:
            logger.debug("[LocalTrace] route=%s latency=%.0fms", post_route, latency_ms)
            return

        try:
            result = pipeline_result or {}
            faith  = result.get("faithfulness") or {}
            docs   = result.get("docs") or []

            with self.client.start_as_current_observation(
                name     = "workflow",
                as_type  = "chain",
                input    = question,
                output   = result.get("answer"),
                metadata = {
                    "session_id":  session_id,
                    "status":      result.get("status"),
                    "confidence":  result.get("confidence"),
                    "route":       post_route,
                    "retry_count": retry_count,
                    "latency_ms":  latency_ms,
                },
            ):
                with self.client.start_as_current_observation(
                    name     = "guardian",
                    as_type  = "guardrail",
                    input    = question,
                    output   = guardian.get("reason", "safe"),
                    metadata = guardian,
                ):
                    pass

                with self.client.start_as_current_observation(
                    name     = "router",
                    as_type  = "span",
                    input    = question,
                    output   = post_route,
                    metadata = {"pre_route": pre_route, "post_route": post_route},
                ):
                    pass

                with self.client.start_as_current_observation(
                    name     = "retrieval",
                    as_type  = "retriever",
                    input    = question,
                    output   = f"{len(docs)} docs retrieved",
                    metadata = {"n_docs": len(docs)},
                ):
                    pass

                with self.client.start_as_current_observation(
                    name     = "conformal",
                    as_type  = "evaluator",
                    input    = {"score": result.get("score"), "threshold": result.get("threshold")},
                    output   = result.get("status", "unknown"),
                    metadata = {
                        "score":      result.get("score"),
                        "threshold":  result.get("threshold"),
                        "confidence": result.get("confidence"),
                    },
                ):
                    pass

                if faith and faith.get("label"):
                    with self.client.start_as_current_observation(
                        name     = "faithfulness",
                        as_type  = "evaluator",
                        input    = result.get("answer", ""),
                        output   = faith.get("label"),
                        metadata = {
                            "label": faith.get("label"),
                            "score": faith.get("faithfulness_score"),
                        },
                    ):
                        pass

            self.client.flush()
            logger.debug("✅ Langfuse trace 发送完成")

        except Exception as e:
            logger.warning("Langfuse trace 发送失败: %s", e)
