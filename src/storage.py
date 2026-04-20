"""
src/storage.py
--------------
SQLite 持久化存储，记录每次问答的完整信息。

功能：
1. 每次问答写入数据库（问题、答案、置信度、faithfulness、来源）
2. 重启后历史不丢失
3. 提供查询接口（按 session、按时间、按状态）
4. 统计接口（幻觉率、拒答率、平均置信度）

应用场景：企业知识库问答的审计日志，
管理员可查看哪些问题回答了、哪些拒答了、幻觉率多少。
"""
from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QAStorage:
    """问答记录持久化存储。"""

    def __init__(self, db_path: str = "data/qa_history.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("QAStorage initialized at %s", db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_records (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT    NOT NULL,
                    timestamp       TEXT    NOT NULL,
                    question        TEXT    NOT NULL,
                    answer          TEXT,
                    status          TEXT    NOT NULL,
                    confidence      REAL,
                    score           REAL,
                    threshold       REAL,
                    alpha           REAL,
                    source          TEXT,
                    faithfulness_score  REAL,
                    faithfulness_label  TEXT,
                    latency_ms      REAL,
                    docs            TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON qa_records(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON qa_records(timestamp)
            """)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, session_id: str, result: dict) -> int:
        """保存一条问答记录，返回记录 ID。"""
        faith = result.get("faithfulness") or {}
        docs  = result.get("docs") or []

        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO qa_records (
                    session_id, timestamp, question, answer,
                    status, confidence, score, threshold, alpha,
                    source, faithfulness_score, faithfulness_label,
                    latency_ms, docs
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                session_id,
                datetime.now().isoformat(),
                result.get("query") or result.get("question", ""),
                result.get("answer"),
                result.get("status", "unknown"),
                result.get("confidence"),
                result.get("score"),
                result.get("threshold"),
                result.get("alpha"),
                result.get("source"),
                faith.get("faithfulness_score"),
                faith.get("label"),
                result.get("latency_ms"),
                json.dumps(docs[:3]) if docs else None,
            ))
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> List[dict]:
        """获取某个 session 的全部记录。"""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_records WHERE session_id=? ORDER BY id",
                (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, limit: int = 20) -> List[dict]:
        """获取最近 N 条记录。"""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM qa_records ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """
        全局统计：
        - 总问答数、已回答数、拒答数
        - 平均置信度
        - Faithfulness 分布
        - 拒答率、幻觉率
        """
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM qa_records"
            ).fetchone()[0]

            answered = conn.execute(
                "SELECT COUNT(*) FROM qa_records WHERE status='reliable'"
            ).fetchone()[0]

            abstained = conn.execute(
                "SELECT COUNT(*) FROM qa_records WHERE status='abstained'"
            ).fetchone()[0]

            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM qa_records WHERE status='reliable'"
            ).fetchone()[0]

            faith_counts = dict(conn.execute("""
                SELECT faithfulness_label, COUNT(*)
                FROM qa_records
                WHERE faithfulness_label IS NOT NULL
                GROUP BY faithfulness_label
            """).fetchall())

            avg_faith = conn.execute(
                "SELECT AVG(faithfulness_score) FROM qa_records "
                "WHERE faithfulness_score IS NOT NULL"
            ).fetchone()[0]

        n_faith_total = sum(faith_counts.values()) or 1
        return {
            "total":            total,
            "answered":         answered,
            "abstained":        abstained,
            "abstention_rate":  round(abstained / total, 4) if total else 0,
            "avg_confidence":   round(avg_conf or 0, 4),
            "faithfulness": {
                "faithful":     faith_counts.get("faithful", 0),
                "uncertain":    faith_counts.get("uncertain", 0),
                "hallucinated": faith_counts.get("hallucinated", 0),
                "faithful_rate":    round(
                    faith_counts.get("faithful", 0) / n_faith_total, 4),
                "hallucinated_rate": round(
                    faith_counts.get("hallucinated", 0) / n_faith_total, 4),
                "avg_score":    round(avg_faith or 0, 4),
            },
        }
