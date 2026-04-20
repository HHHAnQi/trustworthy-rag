# Trustworthy RAG QA

> 基于 Conformal Prediction 的可信 Agentic 问答系统

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)](https://langchain-ai.github.io/langgraph)
[![Langfuse](https://img.shields.io/badge/Langfuse-v4-purple)](https://langfuse.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目背景

现有 RAG（Retrieval-Augmented Generation）系统存在两个核心问题：

1. **过度自信**：模型在知识库覆盖不足时仍会生成答案，导致幻觉（Hallucination）
2. **缺乏量化保证**：无法对"这个答案有多可靠"给出统计上的保证

本项目将 **Conformal Prediction（CP）** 引入 RAG 系统，通过统计方法为拒答决策提供严格的覆盖率保证（miscoverage rate ≤ α），并结合 NLI Faithfulness 检测与 Agentic workflow，构建安全可审计的可信问答系统。

该项目源于硕士研究课题（LLM 不确定性量化），将学术方法工程化落地为完整可用系统。

---

## 核心思想

**宁可不答，不答错。**

```
传统 RAG：问题 → 检索 → 生成 → 返回答案（不管对不对）

本系统：问题 → 检索 → 生成 → CP 评估置信度
                              ├── 置信度高 → 返回答案 + Faithfulness 标签
                              └── 置信度低 → 拒答，告知用户不确定
```

---

## 系统架构

### 整体流程

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────────────┐
│                   Agentic Workflow                   │
│                                                     │
│  Guardian ──► Router ──► Pipeline ──► Logger        │
│  (安全检测)   (路由决策)  (核心推理)  (审计落盘)       │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│                  Pipeline 核心内核                   │
│                                                     │
│  RAG 检索（FAISS top-3）                             │
│     │                                               │
│  LLM 生成（Qwen2.5-7B，K=5 独立采样）                │
│     │                                               │
│  双信号置信度计算                                    │
│  ├── 答案自一致性（K=5 采样的语义相似度）             │
│  └── 检索支持度（答案与文档的 embedding 相似度）      │
│     │                                               │
│  Conformal Prediction 决策                          │
│  ├── score ≤ q̂ → reliable（回答）                   │
│  └── score > q̂ → abstained（触发扩展检索或拒答）     │
│     │                                               │
│  NLI Faithfulness 检测（DeBERTa-v3-small）           │
│  └── faithful / uncertain / hallucinated            │
└─────────────────────────────────────────────────────┘
   │
   ▼
Langfuse 全链路追踪
```

### 四层决策机制

| 层级 | 组件 | 功能 | 实现 |
|------|------|------|------|
| 第一层 | Guardian | 输入安全检测，过滤 prompt 注入 / 越权请求 | `src/guardian.py` |
| 第二层 | Router | 前置路由（clarify/safe_refuse/proceed）+ 后置路由（answer/retrieve_more/abstain） | `src/router.py` |
| 第三层 | CP + NLI | Split CP 统计保证拒答 + NLI 幻觉检测 | `src/conformal.py` + `src/uncertainty.py` |
| 第四层 | Logger + Langfuse | JSONL 本地审计 + Langfuse 云端全链路追踪 | `src/logger.py` + `src/tracer.py` |

---

## Conformal Prediction 原理

### 为什么用 CP

传统置信度分数（如 softmax 输出）没有统计保证，无法回答"这个置信度意味着什么错误率"。

Conformal Prediction 提供严格的覆盖率保证：

> 对于任意 α ∈ (0,1)，在 exchangeability 条件成立时，系统在已回答的问题上，miscoverage rate（错误率）≤ α。

### 实现细节

**Step 1：双信号 nonconformity score**

```python
consistency      = 语义相似度矩阵均值（K=5 采样，屏蔽对角线）
retrieval_support = cosine(answer_embedding, docs_embedding).clip(0,1)
confidence       = w1 * consistency + w2 * retrieval_support
score            = 1 - confidence   # nonconformity score
```

**Step 2：Split CP 校准**

```python
# 在 500 条校准集上计算分位数阈值
q̂ = quantile(cal_scores, ceil((n+1)*(1-α)) / n)
```

**Step 3：推理时决策**

```python
if score <= q̂:
    return answer   # reliable
else:
    return abstain  # 置信度不足，拒答
```

**校准验证**：校准集与测试集 score 分布高度一致（mean 0.302 vs 0.289），验证 exchangeability 条件成立。

---

## 评估结果

在 TriviaQA 数据集上的评估结果（200 条独立测试集）：

| 指标 | 数值 | 说明 |
|------|------|------|
| 无 CP 基线准确率 | 19.5% | 直接回答所有问题 |
| CP 后已回答问题准确率 | **78.0%** | 只回答置信度高的问题 |
| 准确率提升 | **+58.5 pp** | CP 有效过滤低质量回答 |
| Abstention rate | 75% | 系统主动拒绝不确定问题 |
| Faithfulness（faithful） | **93.2%** | 已回答问题中无幻觉比例 |
| Faithfulness（hallucinated） | 4.5% | 已回答问题中幻觉比例 |
| Calibration/Test score 均值 | 0.302 vs 0.289 | exchangeability ✅ |
| 端到端延迟（RTX 3090） | 2-3s | 包含检索、生成、CP、NLI |

---

## 技术栈

```
Python 3.10
FastAPI          — REST API 服务
LangGraph        — 两阶段检索状态机
MCP (fastmcp)    — Tool Server 封装
FAISS            — 向量索引与检索
Qwen2.5-7B-Instruct — 本地 LLM 推理
Transformers     — 模型加载与推理
sentence-transformers — 文本 embedding
Conformal Prediction  — 统计风险控制
DeBERTa-v3-small — NLI Faithfulness 检测
Langfuse v4      — 全链路可观测性
SQLite           — 对话历史持久化
```

---

## 项目结构

```
trustworthy-rag/
├── src/
│   ├── pipeline.py        # RAG + CP + NLI 核心内核（不依赖外部编排）
│   ├── workflow.py        # Agentic 编排层（Guardian + Router + Logger）
│   ├── guardian.py        # 输入安全检测（prompt 注入过滤）
│   ├── router.py          # 有限状态路由器（规则决策，不调用 LLM）
│   ├── logger.py          # JSONL 全链路审计日志
│   ├── tracer.py          # Langfuse v4 追踪封装
│   ├── agent.py           # LangGraph 两阶段检索状态机
│   ├── conformal.py       # Split Conformal Prediction 实现
│   ├── uncertainty.py     # 双信号置信度计算（consistency + retrieval_support）
│   ├── rag.py             # FAISS IndexFlatIP 向量检索
│   ├── llm.py             # Qwen2.5 推理封装（K 次独立采样）
│   ├── mcp_server.py      # MCP Tool Server（trustworthy_qa + extended_search）
│   ├── storage.py         # SQLite 对话历史持久化
│   ├── memory.py          # 多轮对话记忆管理
│   ├── rewriter.py        # Query rewriting（abstained 时改写重试）
│   ├── searcher.py        # 扩展检索（top-10 本地知识库）
│   └── api/
│       └── main.py        # FastAPI 端点定义
├── scripts/
│   ├── build_index.py        # 构建 FAISS 向量索引
│   ├── calibrate.py          # CP 离线校准
│   ├── run_all_and_split.py  # 完整评估（同批数据 cal/test 划分）
│   └── evaluate.py           # 独立评估脚本
├── frontend/
│   └── index.html            # 可视化前端（暖白背景 + 衬线大字答案）
├── configs/
│   └── config.yaml           # 系统配置（模型路径、检索参数、CP 参数）
├── data/
│   ├── qhat.json             # 校准结果（q̂ 阈值表）
│   ├── eval_results.json     # 评估结果
│   └── all700.json           # TriviaQA 评估数据集
└── requirements.txt
```

---

## 快速开始

### 1. 环境准备

```bash
conda create -n trag python=3.10
conda activate trag
pip install -r requirements.txt
```

### 2. 模型下载

需要下载以下模型到 `models/` 目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 主模型（LLM）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/Qwen2.5-7B-Instruct

# 检索 Embedding 模型
huggingface-cli download BAAI/bge-small-en-v1.5 \
    --local-dir models/bge-small-en-v1.5

# NLI Faithfulness 检测模型
huggingface-cli download cross-encoder/nli-deberta-v3-small \
    --local-dir models/nli-deberta-v3-small
```

### 3. 准备知识库

准备 `data/docs.txt`，每行一段文本（Wikipedia 段落或自定义文档），然后构建索引：

```bash
python scripts/build_index.py --config configs/config.yaml
```

### 4. CP 校准

```bash
# 准备 700 条标注数据（question + answer 格式）
python scripts/run_all_and_split.py \
    --config    configs/config.yaml \
    --data_path data/all700.json \
    --cal_size  500 \
    --alphas    0.05 0.1 0.2 \
    --qhat_out  data/qhat.json \
    --eval_out  data/eval_results.json
```

### 5. 启动服务

```bash
# 主 API（端口 6006）
uvicorn src.api.main:app --host 0.0.0.0 --port 6006

# MCP Tool Server（可选，端口 8001）
python src/mcp_server.py
```

浏览器访问 `http://localhost:6006`

### 6. 配置 Langfuse（可选）

在 `configs/.env` 中配置：

```env
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

---

## API 端点

| 端点 | 方法 | 模式 | 说明 |
|------|------|------|------|
| `/ask` | POST | 单轮 | 直接调用 pipeline，延迟最低（2-3s） |
| `/chat` | POST | 多轮 | LangGraph 状态机，支持上下文追问 |
| `/workflow` | POST | Agentic | Guardian + Router + CP + Logger 完整流程 |
| `/recalibrate` | POST | 管理 | 上传新标注数据动态更新 q̂，无需重启 |
| `/history/stats` | GET | 管理 | 全局统计：拒答率、幻觉率、平均置信度 |
| `/history/trace_stats` | GET | 管理 | Workflow 路由分布、Guardian 拦截率、P95 延迟 |
| `/history/recent` | GET | 管理 | 最近 N 条问答记录 |
| `/calibration` | GET | 管理 | 当前 q̂ 阈值表 |
| `/health` | GET | 监控 | 服务健康检查 |

### 请求示例

**单轮问答（/ask）：**
```json
POST /ask
{
    "query": "Who wrote The Phantom of the Opera?",
    "alpha": 0.1
}
```

**Agentic Workflow（/workflow）：**
```json
POST /workflow
{
    "query": "What are the complications of diabetes?",
    "alpha": 0.1,
    "session_id": "user_001"
}
```

**返回结果示例：**
```json
{
    "answer": "Complications include cardiovascular disease, kidney damage...",
    "status": "reliable",
    "confidence": 0.8732,
    "score": 0.1268,
    "threshold": 0.4265,
    "route": "answer",
    "faithfulness": {
        "label": "faithful",
        "score": 0.9991
    },
    "guardian": {"is_safe": true, "reason": "safe"},
    "latency_ms": 2643.9
}
```

---

## 前端功能

浏览器访问 `http://localhost:6006` 后可以：

- **Chat / Workflow 模式切换**：左侧面板切换，底部描述实时更新
- **α 动态调节**：0.05 / 0.10 / 0.20 三档风险等级，实时更新 threshold
- **Faithfulness badge**：绿色 faithful · 99% / 黄色 uncertain / 红色 hallucinated
- **路由状态展示**：
  - ✓ Reliable — answered（绿色）
  - ⚠ Clarify — question too vague（黄色）
  - ✗ Blocked — unsafe input（红色）
  - ✗ Abstained — uncertain（红色）
- **Score vs Threshold 可视化**：直观展示 CP 决策依据
- **Admin Panel**：系统统计、最近记录、在线重校准

---

## 两种使用模式对比

| 特性 | Chat 模式 | Workflow 模式 |
|------|-----------|---------------|
| 多轮对话记忆 | ✅ | ❌（每次独立） |
| Guardian 安全检测 | ❌ | ✅ |
| 模糊问题识别（clarify） | ❌ | ✅ |
| 全链路 Logger | ❌ | ✅ |
| Langfuse 追踪 | ❌ | ✅ |
| 适用场景 | 交互式问答 | 生产审计 |

---

## MCP Tool Server

系统将 RAG+CP pipeline 封装为标准 MCP Tool，支持外部 Agent 通过 MCP 协议接入：

```python
# 启动 MCP Server
python src/mcp_server.py  # 监听 8001 端口

# 外部调用示例
from fastmcp import Client

async with Client("http://localhost:8001/mcp") as client:
    result = await client.call_tool(
        "trustworthy_qa",
        {"question": "Who wrote Hamlet?", "alpha": 0.1}
    )
    print(result.data["answer"])    # "William Shakespeare"
    print(result.data["status"])    # "reliable"
    print(result.data["confidence"]) # 0.87
```

可用工具：
- `trustworthy_qa`：带 CP 置信度保证的问答
- `extended_search`：扩展检索（top-10）

---

## 设计决策

**为什么用 Split CP 而不是 Full CP？**

Full CP 需要在每次推理时重新计算所有校准样本的 score，计算复杂度 O(n)。Split CP 将校准集和测试集分开，只需离线计算一次 q̂，推理时 O(1) 查表，更适合生产部署。

**为什么选 K=5 独立采样？**

采样数越多，self-consistency 估计越准确，但延迟线性增加。K=5 在实验中是延迟（2-3s）和置信度估计质量的最优平衡点。

**为什么 Router 用规则而不是 LLM？**

规则路由延迟 < 1ms，LLM 路由需要额外推理时间（500ms+）。对于 clarify（问题太短/模糊）和 safe_refuse（Guardian 已判定不安全）这类结构性判断，规则完全够用，不需要 LLM。

**pipeline.py 和 workflow.py 为什么分开？**

`pipeline.py` 是纯粹的可信问答内核（RAG+CP+NLI），`workflow.py` 是 Agentic 编排层（Guardian+Router+Logger）。两者分离保证：离线评估（`evaluate.py`）和在线服务走同一套核心逻辑，避免线上线下不一致。

---

## 相关工作

- **Conformal Prediction**：Vovk et al., "Algorithmic Learning in a Random World" (2005)
- **ConU**：Conformal Uncertainty in RAG，本项目不确定性估计的主要参考
- **Self-RAG**：Asai et al., 2024 — 自适应检索的参考方向
- **DeBERTa**：He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
- **Membox**：Tao et al., 2026 — 长期记忆管理参考

---

## License

MIT License

---

## 作者

硕士研究方向：LLM 不确定性量化  
相关论文：基于 Conformal Prediction 的 LLM 输出可靠性评估（在投）
