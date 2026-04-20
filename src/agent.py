"""
src/agent.py
------------
LangGraph 状态机。

流程：
    retrieve（RAG+CP）
        ↓
      judge
      ↙   ↘
reliable  abstained（第一次）
    ↓           ↓
   END       web_search（MCP Tool）
                 ↓
             retrieve_with_search
                 ↓
               judge
              ↙    ↘
         reliable  abstained（第二次）
             ↓           ↓
            END      final_abstain
                         ↓
                        END

answer 字段包含 source 标注：
  "kb"     — 来自本地知识库
  "web"    — 来自网络搜索补充
  "none"   — 最终拒答
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.pipeline import TrustworthyRAGPipeline
from src.rewriter import QueryRewriter
from src.memory import ConversationMemory, Turn

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRY = 1   # 最多触发一次 web search


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    original_question: str
    alpha:             float
    current_question:  str
    search_count:      int        # 已触发 web search 次数
    extra_docs:        List[str]  # web search 返回的文档
    answer:            Optional[str]
    status:            str
    source:            str        # "kb" | "web" | "none"
    confidence:        float
    score:             float
    threshold:         float
    docs:              List[str]
    search_queries:    List[str]  # 记录每次搜索的 query
    faithfulness:      Optional[dict]  # NLI faithfulness 检测结果


# ── Agent ─────────────────────────────────────────────────────────────────────

class TrustworthyAgent:

    def __init__(
        self,
        pipeline:   TrustworthyRAGPipeline,
        rewriter:   QueryRewriter,
        mcp_client  = None,
        searcher    = None,       # WebSearcher 实例，外部注入
    ):
        self.pipeline   = pipeline
        self.rewriter   = rewriter
        self.mcp_client = mcp_client
        self.searcher   = searcher
        self.graph      = self._build_graph()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """RAG + CP，支持注入额外文档（web search 结果）。"""
        question   = state["current_question"]
        alpha      = state["alpha"]
        extra_docs = state["extra_docs"]

        logger.info("Retrieve node — question=%r extra_docs=%d",
                    question, len(extra_docs))

        # 如果有 web search 结果，注入到 pipeline
        if extra_docs:
            # 临时修改 RAG top_k 使 extra_docs 生效
            result = self._run_with_extra_docs(question, alpha, extra_docs)
        else:
            result = self.pipeline.run(
                query=question, alpha=alpha, return_details=True
            )

        # 空答案强制 abstain
        answer = result["answer"]
        if not answer or not answer.strip():
            answer = None
            result["status"] = "abstained"
            result["score"]  = 1.0

        state["answer"]     = answer
        state["status"]     = result["status"]
        state["confidence"] = result["confidence"]
        state["score"]      = result["score"]
        state["threshold"]  = result["threshold"]
        state["docs"]        = result.get("docs", [])
        state["faithfulness"] = result.get("faithfulness")
        return state

    def _node_judge(self, state: AgentState) -> str:
        """条件边：决定下一个节点。"""
        if state["status"] == "reliable":
            logger.info("Judge: reliable, search_count=%d", state["search_count"])
            return "reliable"

        if state["search_count"] >= MAX_SEARCH_RETRY:
            logger.info("Judge: abstained after web search → final_abstain")
            return "give_up"

        logger.info("Judge: abstained → web_search")
        return "search"

    def _node_web_search(self, state: AgentState) -> AgentState:
        """调用 Web Search（通过 MCP client 或直接调用 searcher）。"""
        question = state["original_question"]

        # 用 query rewriter 生成更好的搜索 query
        search_query = self.rewriter.rewrite(question)
        state["search_queries"].append(search_query)
        logger.info("Web search: %r", search_query)

        # 通过 MCP client 调用（如果有），否则直接调用本地 searcher
        if self.mcp_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(
                    self._mcp_web_search(search_query)
                )
            except Exception as e:
                logger.warning("MCP web_search failed: %s, fallback to local", e)
                results = self._local_web_search(search_query)
        else:
            results = self._local_web_search(search_query)

        state["extra_docs"]   = results
        state["search_count"] += 1
        logger.info("Web search returned %d snippets", len(results))
        return state

    async def _mcp_web_search(self, query: str) -> List[str]:
        """通过 MCP 协议调用 web_search tool。"""
        result = await self.mcp_client.call_tool(
            "web_search", {"query": query}
        )
        # fastmcp 返回的是 tool result content
        if isinstance(result, list):
            return result
        if hasattr(result, "content"):
            import json
            try:
                return json.loads(result.content[0].text)
            except Exception:
                return []
        return []

    def _local_web_search(self, query: str) -> List[str]:
        """使用注入的 searcher 做扩展检索。"""
        if self.searcher:
            return self.searcher.search(query)
        from src.searcher import WebSearcher
        fallback = WebSearcher(max_results=10, rag=self.pipeline.rag)
        return fallback.search(query)

    def _node_final_abstain(self, state: AgentState) -> AgentState:
        """最终拒答。"""
        state["answer"] = None
        state["status"] = "abstained"
        state["source"] = "none"
        logger.info("Final abstain for: %r", state["original_question"])
        return state

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("retrieve",      self._node_retrieve)
        g.add_node("web_search",    self._node_web_search)
        g.add_node("final_abstain", self._node_final_abstain)

        g.set_entry_point("retrieve")

        g.add_conditional_edges(
            "retrieve",
            self._node_judge,
            {
                "reliable":    END,
                "search":      "web_search",
                "give_up":     "final_abstain",
            },
        )

        g.add_edge("web_search",    "retrieve")
        g.add_edge("final_abstain", END)

        return g.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        alpha:    float = 0.1,
        memory:   Optional[ConversationMemory] = None,
    ) -> Dict[str, Any]:

        # 多轮对话：注入历史 context
        effective_question = question
        if memory:
            prefix = memory.build_context_prefix()
            if prefix:
                effective_question = prefix + "Current question: " + question

        init_state: AgentState = {
            "original_question": question,
            "alpha":             alpha,
            "current_question":  effective_question,
            "search_count":      0,
            "extra_docs":        [],
            "answer":            None,
            "status":            "abstained",
            "source":            "none",
            "confidence":        0.0,
            "score":             1.0,
            "threshold":         0.0,
            "docs":              [],
            "search_queries":    [],
            "faithfulness":      None,
        }

        final_state = self.graph.invoke(init_state)

        # 根据最终状态推断来源
        if final_state["status"] == "reliable":
            source = "extended_kb" if final_state["search_count"] > 0 else "kb"
        else:
            source = "none"

        result = {
            "question":       question,
            "answer":         final_state["answer"],
            "status":         final_state["status"],
            "source":         source,
            "confidence":     final_state["confidence"],
            "score":          final_state["score"],
            "threshold":      final_state["threshold"],
            "alpha":          alpha,
            "docs":           final_state["docs"],
            "search_queries": final_state["search_queries"],
            "web_searched":   final_state["search_count"] > 0,
            "faithfulness":   final_state.get("faithfulness"),
        }

        if memory is not None:
            memory.add(Turn(
                question   = question,
                answer     = result["answer"],
                status     = result["status"],
                confidence = result["confidence"],
                rewritten  = (final_state["search_queries"][-1]
                              if final_state["search_queries"] else None),
            ))

        return result

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _run_with_extra_docs(
        self, question: str, alpha: float, extra_docs: List[str]
    ) -> Dict:
        """
        把 web search 结果注入 prompt，重新走 LLM + uncertainty + CP。
        extra_docs 拼接到 RAG 检索结果前面，优先级更高。
        """
        from src.pipeline import _PROMPT_TEMPLATE

        # 正常检索
        rag_docs = self.pipeline.rag.retrieve(question)

        # 合并：web search 结果在前（更相关），RAG 结果补充
        combined_docs = extra_docs + rag_docs
        combined_docs = combined_docs[:5]   # 最多 5 条，避免 context 过长

        context = "\n\n".join(
            f"[{i+1}] {d}" for i, d in enumerate(combined_docs)
        )
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        samples = self.pipeline.llm.generate_samples(
            prompt, num_samples=self.pipeline.num_samples
        )
        confidence   = self.pipeline.uncertainty.compute_confidence(
            samples, combined_docs
        )
        nonconformity = 1.0 - confidence
        decision      = self.pipeline.conformal.predict(nonconformity, alpha=alpha)
        best_answer   = self.pipeline._select_answer(samples)
        best_answer   = self.pipeline._postprocess(best_answer)

        # 空答案强制 abstain（模型说不知道，不管 CP 怎么判）
        final_accepted = decision["accepted"] and bool(best_answer)
        return {
            "answer":     best_answer if final_accepted else None,
            "status":     "reliable" if final_accepted else "abstained",
            "confidence": round(confidence, 4),
            "score":      decision["score"] if final_accepted else 1.0,
            "threshold":  decision["threshold"],
            "docs":       combined_docs,
        }
