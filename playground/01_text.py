from agent_core.llm import LLMClient, estimate_cost_cny, system_user

client = LLMClient()

messages = system_user("你是一个技术解释助手，回答要简洁", "用三句话解释什么是 Agent 的上下文工程")

for temp in (0.01, 0.7, 1.2):
    r = client.chat(messages, temperature=temp)
    print(f"\n--- temperature: {temp} ---")
    print(r.text)
    print(f"model={r.model}  latency={r.latency_ms:.0f}ms finish={r.finish_reason}")
    print(f"tokens: in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
    print(f"cost: {estimate_cost_cny(r.usage, r.model)}")

### ① 注意力偏置。 序列开头的 token 会天然聚集更多注意力（叫 attention sink）。把全局设定放开头，它对整段对话的影响最"重"。

### ② 成本。 这条对你 D17 直接有用：KV cache / prompt caching 是从头匹配前缀的。system prompt 固定不变、永远放开头 → 每次请求都能命中缓存，输入成本能降到 1/10。反过来，如果你把变化的内容（用户问题、检索结果）塞在 system 前面，缓存全废。

# 结论（实测，qwen3.8-flash）：
# 1. 收敛型任务 T=0.01 三次要点一致仅措辞不同，T=1.2 出现语病「系统设计地组织」——高温不止更多样，会引入错误。
# 2. 意外：T=0.01 三次、加 seed=42 三次，输出均不相同 → 本模型/端点无法做到逐字可复现。
# 3. 影响：结构化输出与测试必须靠 Schema 校验 + 语义判定，不能做字符串比对；prompt 迭代要靠 Golden Set 统计 pass rate，单次对比无意义。
