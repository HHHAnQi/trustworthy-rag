"""
src/memory.py
-------------
对话历史管理。
支持多轮对话，自动把历史 QA 对注入 prompt context。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Turn:
    question:   str
    answer:     Optional[str]
    status:     str          # "reliable" | "abstained"
    confidence: float
    rewritten:  Optional[str] = None   # 改写后的问题（如果发生了改写）


class ConversationMemory:
    """
    存储多轮对话历史，提供两个功能：
    1. 把历史 QA 对注入新 prompt，支持追问
    2. 记录每轮的置信度和状态，供前端展示
    """

    def __init__(self, max_turns: int = 5):
        self.turns: List[Turn] = []
        self.max_turns = max_turns

    def add(self, turn: Turn):
        self.turns.append(turn)
        # 只保留最近 max_turns 轮
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def build_context_prefix(self) -> str:
        """
        把历史 QA 对拼成 context 前缀，注入到新问题的 prompt 里。
        只包含 reliable 的历史轮次。
        """
        reliable_turns = [t for t in self.turns if t.status == "reliable"
                          and t.answer]
        if not reliable_turns:
            return ""
        lines = ["Previous conversation:"]
        for t in reliable_turns[-3:]:   # 最多引用最近 3 轮
            lines.append(f"Q: {t.question}")
            lines.append(f"A: {t.answer}")
        return "\n".join(lines) + "\n\n"

    def clear(self):
        self.turns = []

    def to_dict(self) -> list:
        return [
            {
                "question":   t.question,
                "answer":     t.answer,
                "status":     t.status,
                "confidence": t.confidence,
                "rewritten":  t.rewritten,
            }
            for t in self.turns
        ]
