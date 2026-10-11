"""D1 练习 2：流式调用 —— 量出 TTFT（首字延迟）、吞吐，以及**思维链的开销**。

跑法：  uv run python playground/02_stream.py

这个脚本回答四个问题：
  ① 字是不是真的一个一个出来的？（不是"一次性打印"）
  ② 首字延迟（TTFT）和总耗时各是多少？差多少？
  ③ usage 为什么在流式下只能从最后一个分片拿到？
  ④ （实测新增）推理模型的思维链算在哪、要花多少钱？

第 ④ 问是跑通之后才发现的：qwen3.8 是推理模型，它会先输出一段
reasoning_content（思维链）再输出正文。实测「数 1 到 3」这种请求，
64 个分片里 59 个是思维链，最终可见内容只有 3 个字符，但输出 token 记了 78 个。
也就是说：
    · 只统计 content，你会完全看不到成本的大头
    · 只按第一段 content 算 TTFT，你会把思维链的时间误判成"首字慢"
所以这个脚本把两类流分开计时、分开计数、分开算钱。
"""

from __future__ import annotations

import time

from agent_core.llm import (
    PRICES_CNY_PER_1K,
    LLMClient,
    Usage,
    estimate_cost_cny,
    system_user,
)

SYSTEM = "你是一个技术解释助手，回答要简洁但要有信息量。"
QUESTION = "解释什么是 Java 中 HashMap 的底层数据结构"


def main() -> None:
    client = LLMClient()
    messages = system_user(SYSTEM, QUESTION)

    print("=" * 78)
    print(
        f"模型: {client.settings.dashscope_text_model}   provider: {client.settings.llm_provider}"
    )
    print("=" * 78)

    # 计时用 perf_counter（单调时钟）。不要用 time.time()——
    # 它会受系统时间调整影响，测延迟必须用单调时钟。
    started = time.perf_counter()

    first_any_ms: float | None = None  # 第一个非空分片（不论思维链还是正文）
    first_reasoning_ms: float | None = None
    first_content_ms: float | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    reasoning_chars = 0
    content_chars = 0
    n_chunks = 0

    print("\n↓ 注意观察：可能先出现一段「思维链」，再出现正文\n")

    for chunk in client.stream(messages, temperature=0.3):
        n_chunks += 1
        now_ms = (time.perf_counter() - started) * 1000

        # 每个分片都同时检查 reasoning / delta / usage / finish_reason 四个东西。
        # 不要假设"第几个分片是什么"——不同 provider 的切法不一样。
        if chunk.reasoning:
            if first_any_ms is None:
                first_any_ms = now_ms
            if first_reasoning_ms is None:
                first_reasoning_ms = now_ms
                print("\n[思维链开始] ", end="", flush=True)
            # 思维链通常很长，这里只打进度点，避免刷屏；
            # 真实产品里这一步是"思考中…"的展示位。
            reasoning_chars += len(chunk.reasoning)

        if chunk.delta:
            if first_any_ms is None:
                first_any_ms = now_ms
            if first_content_ms is None:
                first_content_ms = now_ms
                print("\n\n[正文开始] ", end="", flush=True)
            content_chars += len(chunk.delta)
            # end="" 不换行；flush=True 强制立即写出。
            # 少了 flush，Python 按行缓冲，流式效果会被自己吃掉。
            print(chunk.delta, end="", flush=True)

        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    total_ms = (time.perf_counter() - started) * 1000
    print("\n")

    if first_any_ms is None:
        print("没有收到任何内容，先检查模型名和额度")
        return

    # ------------------------------------------------------------------
    # 延迟分解
    # ------------------------------------------------------------------
    print("=" * 78)
    print("延迟分解")
    print("=" * 78)
    print(f"  排队 + prefill + 网络   : {first_any_ms:9.0f} ms  <- 第一个分片到达")
    if first_reasoning_ms is not None:
        print(
            f"  思维链阶段              : "
            f"{(first_content_ms or total_ms) - first_reasoning_ms:9.0f} ms"
            "  <- 用户此时看不到任何正文"
        )
    if first_content_ms is not None:
        print(f"  首字延迟 TTFT(正文)     : {first_content_ms:9.0f} ms  <- 用户真正看到字的时刻")
        print(f"  正文生成阶段            : {total_ms - first_content_ms:9.0f} ms")
    print(f"  总耗时                  : {total_ms:9.0f} ms")
    print(f"  分片总数                : {n_chunks:9d}")
    print(f"  思维链字符数            : {reasoning_chars:9d}")
    print(f"  正文字符数              : {content_chars:9d}")

    if first_content_ms:
        print(
            f"\n  注意：如果把 TTFT 定义成「第一个分片」({first_any_ms:.0f} ms) 或"
            f"「第一段正文」({first_content_ms:.0f} ms)，\n"
            f"  两个数字差 {first_content_ms - first_any_ms:.0f} ms，"
            "对应的用户体验完全不同。**指标定义比指标本身更重要。**"
        )

    # ------------------------------------------------------------------
    # 用量与成本分解
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("用量与成本分解")
    print(f"{'=' * 78}")
    if usage is None:
        print("  没拿到 usage。检查 stream() 是否传了")
        print("  stream_options={'include_usage': True}（部分 provider 不支持该参数）")
    else:
        print(f"  输入 tokens             : {usage.prompt_tokens:9d}")
        print(f"  输出 tokens（合计）      : {usage.completion_tokens:9d}")
        print(f"    其中思维链            : {usage.reasoning_tokens:9d}")
        print(f"    其中可见正文          : {usage.visible_tokens:9d}")

        if usage.completion_tokens and usage.reasoning_tokens:
            ratio = usage.completion_tokens / max(1, usage.visible_tokens)
            print(
                f"\n  思维链把输出放大了 {ratio:.1f} 倍：你为 {usage.visible_tokens} 个可见 token"
                f"付了 {usage.completion_tokens} 个 token 的钱。"
            )
        elif usage.completion_tokens and not usage.reasoning_tokens:
            print("\n  provider 未返回 reasoning_tokens 明细，无法拆分思维链成本。")

        gen_ms = total_ms - (first_any_ms or 0)
        if gen_ms > 0 and usage.completion_tokens:
            print(
                f"  生成吞吐（含思维链）     : {usage.completion_tokens / (gen_ms / 1000):9.1f} tok/s"
            )

        cost = estimate_cost_cny(usage, client.settings.dashscope_text_model)
        if cost is None:
            print("  本次成本                : 未知（该模型未登记价格）")
        else:
            print(f"  本次成本                : {cost:9.6f} 元")
            if usage.reasoning_tokens:
                # 价格从 PRICES_CNY_PER_1K 读，不要在这里再写一遍——
                # 硬编码的价格副本一定会和 config 里的那张表失去同步。
                price = PRICES_CNY_PER_1K.get(client.settings.dashscope_text_model)
                if price:
                    r_cost = usage.reasoning_tokens / 1000 * price[1]
                    print(
                        f"    其中思维链成本        : {r_cost:9.6f} 元"
                        f"  ({r_cost / cost * 100:.0f}%)\n"
                    )

    print(f"  结束原因                : {finish_reason}")
    if finish_reason == "length":
        print("  ⚠️ 被 max_tokens 截断。对结构化输出而言这必须当成一次失败处理。")
    elif finish_reason == "tool_calls":
        print("  → 模型要求调用工具。D3 手写 Agent 循环时，这就是循环继续的分支条件。")
    print()


if __name__ == "__main__":
    main()
