"""
src/logger.py
-------------
全链路 JSONL trace logger。

每次请求记录一行 JSON，包含：
  - 请求信息（session_id, question, timestamp）
  - Guardian 结果
  - 路由决策（pre_route, post_route）
  - 检索信息（n_docs, top_score）
  - CP 评估（score, threshold, status）
  - Faithfulness（label, score）
  - 最终结果（answer, latency）

第一版用 JSONL 文件，易 debug、易回放。
后续可扩展接入 SQLite / Langfuse。
"""
from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    JSONL 格式的全链路追踪日志。
    每行一个 JSON 对象，对应一次完整请求。
    """

    def __init__(self, log_path: str = "logs/trace.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("TraceLogger 初始化: %s", self.log_path)

    def log(
        self,
        session_id:  str,
        question:    str,
        guardian:    Dict[str, Any],
        pre_route:   str,
        post_route:  str,
        pipeline_result: Dict[str, Any],
        retry_count: int,
        latency_ms:  float,
        extra:       Optional[Dict] = None,
    ) -> None:
        """
        记录一次完整请求的全链路信息。
        """
        result = pipeline_result or {}
        breakdown = result.get("breakdown") or {}
        faith = result.get("faithfulness") or {}

        record = {
            # 基础信息
            "timestamp":   datetime.now().isoformat(),
            "session_id":  session_id,
            "question":    question,

            # 安全层
            "guardian": {
                "is_safe": guardian.get("is_safe", True),
                "reason":  guardian.get("reason", ""),
            },

            # 路由决策
            "routing": {
                "pre_route":   pre_route,
                "post_route":  post_route,
                "retry_count": retry_count,
            },

            # 检索信息
            "retrieval": {
                "n_docs":     len(result.get("docs") or []),
                "top_score":  breakdown.get("retrieval_support"),
            },

            # CP 评估
            "conformal": {
                "score":     result.get("score"),
                "threshold": result.get("threshold"),
                "alpha":     result.get("alpha"),
                "status":    result.get("status"),
                "confidence": result.get("confidence"),
            },

            # Faithfulness
            "faithfulness": {
                "label": faith.get("label"),
                "score": faith.get("faithfulness_score"),
            },

            # 最终结果
            "output": {
                "answer":     result.get("answer"),
                "final_route": post_route,
                "latency_ms": latency_ms,
            },
        }

        if extra:
            record["extra"] = extra

        # 写入 JSONL
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("TraceLogger 写入失败: %s", e)

    def get_recent(self, n: int = 20) -> list:
        """读取最近 n 条 trace 记录。"""
        try:
            lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
            lines = [l for l in lines if l.strip()]
            recent = lines[-n:]
            return [json.loads(l) for l in recent]
        except Exception:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        统计全部 trace 的路由分布和关键指标。
        对应 ChatGPT 方案里的 Route Distribution 指标。
        """
        records = self.get_recent(n=10000)
        if not records:
            return {}

        total = len(records)
        routes = {}
        n_faithful = n_hallucinated = n_blocked = 0
        latencies = []

        for r in records:
            # 路由分布
            route = r.get("routing", {}).get("post_route", "unknown")
            routes[route] = routes.get(route, 0) + 1

            # Guardian 拦截
            if not r.get("guardian", {}).get("is_safe", True):
                n_blocked += 1

            # Faithfulness
            label = r.get("faithfulness", {}).get("label")
            if label == "faithful":
                n_faithful += 1
            elif label == "hallucinated":
                n_hallucinated += 1

            # 延迟
            lat = r.get("output", {}).get("latency_ms")
            if lat:
                latencies.append(lat)

        latencies.sort()
        n_answered = routes.get("answer", 0)

        return {
            "total":           total,
            "route_distribution": {
                k: {"count": v, "rate": round(v/total, 4)}
                for k, v in routes.items()
            },
            "guardian_block_rate": round(n_blocked/total, 4) if total else 0,
            "faithfulness": {
                "faithful_rate":    round(n_faithful/n_answered, 4) if n_answered else 0,
                "hallucinated_rate": round(n_hallucinated/n_answered, 4) if n_answered else 0,
            },
            "latency": {
                "avg_ms": round(sum(latencies)/len(latencies), 1) if latencies else 0,
                "p95_ms": round(latencies[int(len(latencies)*0.95)], 1) if latencies else 0,
            },
        }
