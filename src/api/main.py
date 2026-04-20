"""
src/api/main.py  v2.1
---------------------
POST /ask    单轮（原有）
POST /chat   多轮 Agent（LangGraph + Web Search MCP）
GET  /health
GET  /calibration
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.pipeline import TrustworthyRAGPipeline
from src.agent import TrustworthyAgent
from src.rewriter import QueryRewriter
from src.memory import ConversationMemory
from src.storage import QAStorage
from src.guardian import Guardian
from src.logger   import TraceLogger
from src import router as router_module
from src.workflow import AgenticWorkflow
from src.tracer   import LangfuseTracer as LFTracer
from src.utils import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

config   = load_config("configs/config.yaml")
pipeline = TrustworthyRAGPipeline(config)
rewriter = QueryRewriter(pipeline.llm)
# WebSearcher 注入 RAG 实例，用本地扩展检索替代外网搜索
from src.searcher import WebSearcher
searcher = WebSearcher(max_results=10, rag=pipeline.rag)
agent    = TrustworthyAgent(pipeline, rewriter, mcp_client=None, searcher=searcher)

_sessions: Dict[str, ConversationMemory] = {}
storage = QAStorage(config.get('storage', {}).get('db_path', 'data/qa_history.db'))


app = FastAPI(
    title="Trustworthy RAG QA",
    description="RAG + Conformal Prediction + LangGraph + Web Search MCP",
    version="2.1.0",
)

guardian     = Guardian(pipeline.llm, fail_mode="open")
trace_logger = TraceLogger("logs/trace.jsonl")
workflow     = AgenticWorkflow(pipeline, guardian, router_module, trace_logger)
lf_tracer    = LFTracer()
workflow.langfuse_tracer = lf_tracer
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

if Path("frontend").exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    def index():
        return FileResponse("frontend/index.html")


class AskRequest(BaseModel):
    query:          str             = Field(..., min_length=1)
    alpha:          Optional[float] = Field(None, gt=0, lt=1)
    return_details: bool            = False


class ChatRequest(BaseModel):
    query:      str             = Field(..., min_length=1)
    alpha:      Optional[float] = Field(None, gt=0, lt=1)
    session_id: str             = Field(default="default")


class AskResponse(BaseModel):
    query:         str
    answer:        Optional[str]
    confidence:    float
class WorkflowResponse(BaseModel):
    query:        str
    answer:       Optional[str]
    status:       str
    confidence:   float
    score:        float
    threshold:    float
    alpha:        float
    route:        str
    retry_count:  int
    latency_ms:   float
    docs:         List[str]
    accepted:     bool = False
    faithfulness: Optional[dict] = None
    guardian:     Optional[dict] = None
    message:      Optional[str]  = None




    status:        str
    threshold:     float
    alpha:         float
    score:         float
    latency_ms:    float
    faithfulness:  Optional[dict] = None
    docs:          Optional[list] = None
    samples:       Optional[list] = None
    breakdown:     Optional[dict] = None


class ChatResponse(BaseModel):
    query:          str
    answer:         Optional[str]
    status:         str
    source:         str
    confidence:     float
    score:          float
    threshold:      float
    alpha:          float
    web_searched:   bool
    search_queries: List[str]
    docs:           List[str]
    history:        List[dict]
    latency_ms:     float
    faithfulness:   Optional[dict] = None


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        result = pipeline.run(
            query=req.query, alpha=req.alpha,
            return_details=req.return_details,
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(500, detail=str(exc)) from exc
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    storage.save(session_id="single", result=result)
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    if req.session_id not in _sessions:
        _sessions[req.session_id] = ConversationMemory(max_turns=5)
    memory = _sessions[req.session_id]
    try:
        result = agent.run(
            question = req.query,
            alpha    = req.alpha or config["conformal"]["default_alpha"],
            memory   = memory,
        )
    except Exception as exc:
        logger.exception("Agent error")
        raise HTTPException(500, detail=str(exc)) from exc
    return {
        "query":          req.query,
        "answer":         result["answer"],
        "status":         result["status"],
        "source":         result["source"],
        "confidence":     result["confidence"],
        "score":          result["score"],
        "threshold":      result["threshold"],
        "alpha":          result["alpha"],
        "web_searched":   result["web_searched"],
        "search_queries": result["search_queries"],
        "docs":           result["docs"],
        "history":        memory.to_dict(),
        "latency_ms":     round((time.perf_counter() - t0) * 1000, 1),
        "faithfulness":   result.get("faithfulness"),
    }
    storage.save(session_id=req.session_id, result={
        **result,
        "query": req.query,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    })


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].clear()
    return {"cleared": session_id}



@app.post("/workflow", response_model=WorkflowResponse)
def workflow_endpoint(req: ChatRequest):
    """
    Agentic RAG 端点：Guardian + Router + Pipeline + Logger。
    完整的可信 Agent 流程，离线评测和在线服务走同一套逻辑。
    """
    try:
        result = workflow.run(
            query      = req.query,
            alpha      = req.alpha or config["conformal"]["default_alpha"],
            session_id = req.session_id,
        )
    except Exception as exc:
        logger.exception("Workflow error")
        raise HTTPException(500, detail=str(exc)) from exc
    storage.save(session_id=req.session_id, result={
        "query":      req.query,
        "answer":     result.get("answer"),
        "status":     result.get("status", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "latency_ms": result.get("latency_ms", 0),
    })
    return result


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}


class RecalibrateRequest(BaseModel):
    data:   list  = Field(..., description="校准数据，每条含 question 和 answer")
    alphas: list  = Field(default=[0.05, 0.1, 0.2])


@app.post("/recalibrate", summary="在线重校准 — 上传新数据重新计算 q̂")
def recalibrate(req: RecalibrateRequest):
    """
    管理员接口：上传一批标注数据，系统重新计算 q̂ 并立即生效。
    无需重启服务，解决生产环境数据分布漂移问题。

    请求体示例：
    {
        "data": [
            {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
            ...
        ],
        "alphas": [0.05, 0.1, 0.2]
    }
    """
    from src.conformal import compute_qhat

    if len(req.data) < 50:
        raise HTTPException(
            400,
            detail="校准数据至少需要 50 条，当前只有 %d 条，建议使用 100 条以上" % len(req.data)
        )

    # 1. 对每条数据跑 pipeline，收集 nonconformity score
    scores = []
    failed = 0
    for item in req.data:
        try:
            result = pipeline.run(
                query=item["question"],
                alpha=req.alphas[-1],
                return_details=False,
            )
            score = result["score"]
            # 空答案强制 score=1.0
            if not result.get("answer"):
                score = 1.0
            scores.append(score)
        except Exception:
            failed += 1

    if len(scores) < 10:
        raise HTTPException(
            500,
            detail="有效数据不足（成功=%d，失败=%d），无法校准" % (
                len(scores), failed
            )
        )

    # 2. 重新计算 q̂
    new_qhat = {}
    for alpha in req.alphas:
        new_qhat[str(alpha)] = round(compute_qhat(scores, alpha), 6)

    # 3. 更新内存中的阈值（立即生效，无需重启）
    old_qhat = dict(pipeline.conformal.qhat_table)
    pipeline.conformal.qhat_table = new_qhat

    logger.info(
        "在线重校准完成: n=%d, 旧 q̂=%s, 新 q̂=%s",
        len(scores), old_qhat, new_qhat
    )

    return {
        "status":    "ok",
        "n_used":    len(scores),
        "n_failed":  failed,
        "old_qhat":  old_qhat,
        "new_qhat":  new_qhat,
        "message":   "q̂ 已更新，新阈值立即生效，无需重启服务",
    }


@app.get("/history/trace_stats", summary="Workflow trace 路由分布统计")
def history_trace_stats():
    """从 JSONL trace 文件获取路由分布、Guardian 拦截率、延迟统计。"""
    return trace_logger.get_stats()


@app.get("/history/stats", summary="全局统计：拒答率、幻觉率、平均置信度")
def history_stats():
    """
    企业知识库审计视图：
    查看系统整体表现，包括拒答率、幻觉率、平均置信度。
    """
    return storage.get_stats()


@app.get("/history/recent", summary="最近问答记录")
def history_recent(limit: int = 20):
    """返回最近 N 条问答记录。"""
    return storage.get_recent(limit=limit)


@app.get("/history/session/{session_id}", summary="查询某个 session 的历史")
def history_session(session_id: str):
    """返回指定 session 的全部问答记录。"""
    return storage.get_session(session_id)


@app.get("/calibration")
def calibration():
    return {
        "qhat_table":    pipeline.conformal.qhat_table,
        "default_alpha": pipeline.conformal.default_alpha,
    }
