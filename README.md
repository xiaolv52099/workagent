# WorkAgent

> 把散落在 PDF / 网页 / 表格里的企业文档，变成**可问答、可执行多步任务、且每一步可追溯可评测**的 Agent 工作台。

一个 AI Native 产品，不是 Demo。核心目标是把「Agent 能在真实业务里稳定跑起来」这件事做出来——
所以这个项目的一半代码在解决**编排、状态、失败恢复、权限、成本和评测**，而不是在写 prompt。

---

## 为什么做这个

（这一节是给面试官看的，也是给你自己对齐目标的）

LLM 应用的门槛不是「调通 API」，而是让它在生产环境里可靠地完成多步任务。这个判断不是我猜的，
是 2026 年国内 AI 应用 / Agent 岗位 JD 的共识——合并约 50 条真实 JD 后，出现频率最高的要求是：

| 要求 | 出现率 | 本项目对应模块 |
| --- | --- | --- |
| RAG 全流程（含混合检索 / Rerank） | 84-87% | `agent_core/rag/` |
| Tool / Function Calling + 结构化输出 | 74-87% | `agent_core/tools/` + `schema.py` |
| Prompt / Context Engineering | 68-93% | `prompts/` + 上下文预算管理 |
| 服务化与工程基础（FastAPI / Docker / CI） | 63-87% | `api/` + `Dockerfile` |
| Agent 核心机制（任务规划 / 状态 / 记忆） | 47-60% | `agent_core/orchestrate/` + `memory/` |
| 可观测与评测（Eval / Trace / 成本归因） | 37-67% | `evals/` + `obs/` |
| MCP 工具协议 | 约 47% | `mcp_server/`（并开源） |
| 高并发 / 异步调度 / 消息队列 | 37-60% | 任务队列 + checkpoint |

---

## 技术栈

| 层 | 选型 | 为什么 |
| --- | --- | --- |
| 语言 | Python 3.12（锁版本，不用系统 3.14） | 生态兼容性最好 |
| 模型接入 | OpenAI 兼容协议（百炼 DashScope + 火山方舟 Ark） | 一套 SDK 接多家，换模型不改业务代码 |
| 包管理 | `uv` | 快，且能自动管理 Python 版本 |
| Web | FastAPI | JD 高频出现 |
| 校验 | Pydantic v2 | 结构化输出的地基 |
| 质量 | ruff + mypy(strict) + pytest | 工程规范的底线 |

后续几天会加入：pgvector（检索）、任务队列（异步）、Langfuse（可观测）、LangGraph（编排第二实现）。

---

## 目录结构

```
workagent/
├── README.md
├── pyproject.toml
├── .env.example            # 复制成 .env 再填密钥；.env 不入库
├── src/agent_core/
│   ├── config.py           # 配置与密钥（只从环境变量读）
│   ├── llm.py              # 统一模型接入层：非流式 / 流式 / 视觉 + 用量成本记账
│   └── __init__.py         # 目录约定（10 层内核规划）
├── playground/             # D1 练习：三个最小调用 + token 计数器
├── tests/                  # 单测（不依赖网络）
└── data/                   # 运行时产物，不入库
```

---

## 快速开始

```bash
# 1) 锁定 Python 版本（会用到 3.12，uv 自动下载）
uv python pin 3.12

# 2) 装依赖（含 dev）
uv sync

# 3) 配置密钥
cp .env.example .env
#    然后编辑 .env，填 DASHSCOPE_API_KEY（和可选的 ARK_API_KEY）

# 4) 冒烟测试（不需要密钥，验证环境正常）
uv run pytest -q

# 5) 质量检查
uv run ruff check . && uv run mypy src
```

准备就绪后，按 `playground/README.md` 完成 D1 的四个练习。

---

## 安全约定（项目宪法的一部分）

1. **密钥只走环境变量**，`.env` 永不入库；`.env.example` 里只放占位符。
2. **禁止在业务层拼接 SQL**（D4 接入数据库后生效）。
3. **有副作用的工具必须幂等 + 有审计日志**（D11 接入工具层后生效）。
4. **关键链路（鉴权 / 资金类 / 事务）人工复核**，不交给 AI 定稿。
