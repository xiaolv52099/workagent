"""D1 练习 2：流式调用 —— 量出 TTFT（首字延迟）和吞吐。

跑法：  uv run python playground/02_stream.py

这个脚本要回答三个问题：
  ① 字是不是真的一个一个出来的？（不是"一次性打印"）
  ② 首字延迟（TTFT）和总耗时各是多少？差多少？
  ③ usage 为什么在流式下只能从最后一个分片拿到？
"""

from __future__ import annotations

import time

from agent_core.llm import LLMClient, Usage, estimate_cost_cny, system_user

SYSTEM = "你是一个技术解释助手，回答要简洁但要有信息量。"
QUESTION = "解释什么是 Java 中HashMap 的底层数据结构"


def main() -> None:
    client = LLMClient()
    messages = system_user(SYSTEM, QUESTION)

    print("=" * 72)
    print(
        f"模型: {client.settings.dashscope_text_model}   provider: {client.settings.llm_provider}"
    )
    print("=" * 72)
    print("↓ 下面是流式输出，注意观察：字是一个一个冒出来的，不是整段一次性出现\n")

    # ------------------------------------------------------------------
    # 计时：用 perf_counter（单调时钟），不要用 time.time()。
    # 后者会受系统时间调整影响，测延迟必须用单调时钟。
    # ------------------------------------------------------------------
    started = time.perf_counter()

    ttft_ms: float | None = None  # Time To First Token：从发请求到收到第一个字
    usage: Usage | None = None  # 用量只在最后一个分片里
    finish_reason: str | None = None
    char_count = 0

    for chunk in client.stream(messages, temperature=0.3):
        # 1) 正文增量
        if chunk.delta:
            if ttft_ms is None:
                # 第一个非空分片到达的时刻 = TTFT
                ttft_ms = (time.perf_counter() - started) * 1000
            # end="" 不换行；flush=True 强制立即写出。
            # 少了 flush，Python 会按行缓冲，你看到的就是"憋一大段再一起出来"——
            # 流式的效果被你自己吃掉了。
            print(chunk.delta, end="", flush=True)
            char_count += len(chunk.delta)

        # 2) 用量（只出现在最后一个分片）
        if chunk.usage is not None:
            usage = chunk.usage

        # 3) 结束原因（出现在倒数第二个分片，此时 delta 为空）
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason

    total_ms = (time.perf_counter() - started) * 1000
    print("\n")

    # ------------------------------------------------------------------
    # 指标汇总
    # ------------------------------------------------------------------
    print("=" * 72)
    print("指标")
    print("=" * 72)

    if ttft_ms is None:
        print("没有收到任何内容，先检查模型名和额度")
        return

    gen_ms = total_ms - ttft_ms  # 真正的"生成阶段"，不含排队与首字开销
    print(f"  首字延迟 TTFT   : {ttft_ms:9.0f} ms   <- 用户感觉等了多久")
    print(f"  总耗时          : {total_ms:9.0f} ms")
    print(f"  生成阶段耗时    : {gen_ms:9.0f} ms   <- 总耗时 - TTFT")
    print(f"  输出字符数      : {char_count:9d}")

    if usage is not None:
        print(f"  输入 tokens     : {usage.prompt_tokens:9d}")
        print(f"  输出 tokens     : {usage.completion_tokens:9d}")
        print(f"  合计 tokens     : {usage.total_tokens:9d}")

        # 吞吐 = 输出 token / 生成阶段秒数。
        # 这个数字直接决定用户体验，也是 D18 延迟分解里的一项重要。
        if gen_ms > 0 and usage.completion_tokens:
            tps = usage.completion_tokens / (gen_ms / 1000)
            print(f"  生成吞吐        : {tps:9.1f} tok/s")

        cost = estimate_cost_cny(usage, client.settings.dashscope_text_model)
        print(f"  本次成本        : {cost if cost is None else f'{cost:.6f} 元'}")
    else:
        print("  没拿到 usage。检查 stream() 里是否传了")
        print("  stream_options={'include_usage': True}（部分 provider 不支持该参数）")

    print(f"  结束原因        : {finish_reason}")
    print()


if __name__ == "__main__":
    main()
