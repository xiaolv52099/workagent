"""OpenAI Agents SDK + 阿里云百炼 的最小可用示例。

跑法：  uv run python playground/agent_tool.py

三层报错的完整链条（每一层都是真实的）：

  第 1 层  TypeError: Agent.__init__() got an unexpected keyword argument 'description'
          → Agent 没有 description，对应参数是 instructions。
            （description 是别的框架的写法：CrewAI 用 role/goal/backstory，
             smolagents 里 description 是给 Tool 用的。抄教程时最容易串。）

  第 2 层  openai.OpenAIError: Missing credentials ... OPENAI_API_KEY is not set
          → 把 description 改对之后必然撞上这个。SDK 默认连 OpenAI 官方 API，
            你没有它的 key，国内也访问不到。**必须显式把模型通道指向百炼。**

  第 3 层  切了 base_url 之后还可能 404
          → 新版 SDK 默认走 Responses API，百炼只兼容 chat completions。
            不显式切换就会失败。

所以下面这段"provider 配置"不是样板代码，是必需的。
"""

from __future__ import annotations

import asyncio

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from agent_core.config import get_settings

# ---------------------------------------------------------------------------
# 1) 把模型通道指向百炼（修第 2、3 层报错）
# ---------------------------------------------------------------------------
# 密钥继续走 .env（项目宪法的红线：密钥只从环境变量读），
# 所以通过 config.get_settings() 取，而不是在脚本里硬编码。
settings = get_settings()

client = AsyncOpenAI(
    api_key=settings.dashscope_api_key,
    base_url=settings.dashscope_base_url,  # OpenAI 兼容入口
)

# use_for_tracing=False 很重要：SDK 默认会把这个 client 同时用于
# 向 OpenAI 上报 trace。我们没有那个账号，开着只会在后台刷报错。
set_default_openai_client(client, use_for_tracing=False)

# 修第 3 层：显式声明用 chat_completions 而不是 responses
set_default_openai_api("chat_completions")

# 关掉 tracing 上报。生产环境这里应该换成自己的 Langfuse（见 D18）
set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# 2) 定义 Agent
# ---------------------------------------------------------------------------
agent = Agent(
    name="a helpful assistant",
    # 修第 1 层：原本写的是 description，正确参数是 instructions
    #
    # 指令写法说明：最初这里写的是 "only answer in English"，
    # 但用中文提问时模型照样回中文 —— 实测「模糊软约束会被忽略」。
    # 改成具体、可检测的约束后就稳定生效了。
    # 详见 notes/D1-实测笔记.md 第 10 条。
    instructions=(
        "You are a helpful assistant. "
        "Always reply in English only, regardless of the language the user writes in."
    ),
    # 模型必须显式指定：SDK 的默认值是 OpenAI 的模型名，百炼上没有 -> 404
    model=settings.dashscope_text_model,
)


async def main() -> None:
    result = await Runner.run(agent, "你是谁？")
    print("回答      :", result.final_output)
    # 注意输入 token 是 60+ —— D1 实测过每个请求有约 60 token 的固定开销，
    # 所以哪怕只问一句"你是谁"，成本也不是 0
    print("输入 token:", result.context_wrapper.usage.input_tokens)
    print("输出 token:", result.context_wrapper.usage.output_tokens)


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# 3) 关于"要不要用框架"（这比代码本身重要，值得读完）
# ---------------------------------------------------------------------------
# 你现在的进度：D1 做完了，D2/D3 还没做。而 D3 的作业是
# **手写 ReAct 循环（明确禁止用框架）**。
#
# 为什么计划里这么安排 —— JD 原文写了两句看起来矛盾的话：
#     「有 LangGraph、Agents SDK 等 Agent Framework 的实际项目经验」
#     「但不依赖框架解决问题」
#
# 面试官问的从来不是"你会不会用框架"，而是：
#     「框架内部替你做了什么？去掉框架你会怎么做？」
#
# 所以正确的顺序是：
#     ① 先手写一遍 loop（D3）→ 拿到"没有框架我也能实现"的底气
#     ② 再用 Agents SDK / LangGraph 各写一版 → 拿到"用过框架"的证据
#     ③ 最后写下对比结论 → 这一步才是加分项
#
# 这个文件现在是"② 的一半"：让你看到框架的 API 长什么样。
# 但如果只停在这里，被追问这些就会卡住：
#     · Runner.run 内部循环了几轮？终止条件是什么？
#     · 工具执行结果是以什么 role 拼回 messages 的？
#     · 上下文什么时候会爆掉？爆了怎么处理？
#     · 它怎么处理工具抛异常？
# 这些恰好是 D3 手写一遍才能内化的东西。
#
# 建议：这个已经跑通了 → 回去做 D2（结构化输出）→ D3 手写循环
#       → 再回来给这个 Agent 加 @function_tool，两版对照写笔记。
