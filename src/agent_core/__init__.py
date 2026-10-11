"""agent_core —— WorkAgent 的内核包。

目录约定（后续几周会逐步长成这 10 层）：
    config.py      配置与密钥（只从环境变量读）
    llm.py         统一模型接入层：多 provider、非流式/流式/视觉、用量与成本记账
    prompts/       Prompt 模板与版本（D2）
    schema.py      结构化输出契约（D2）
    tools/         工具注册表 + 权限 + 幂等 + 副作用审计（D11）
    rag/           摄取 → 分块 → embedding → 混合检索 → rerank → 引用（D4-D5）
    orchestrate/   自研 Step 编排引擎 + checkpoint（D9）
    memory/        会话记忆 + 长期偏好记忆（D11）
    evals/         Golden Set + LLM-as-Judge + 回归（D15-D17）
    obs/           可观测：trace、token 成本、延迟分解（D18）
"""

__version__ = "0.1.0"
