# playground —— D1 的三个练习（**这部分你自己写**）

脚手架已经给好了（`agent_core/config.py` 和 `agent_core/llm.py`）。这里的三份脚本是**你的作业**——
面试时会问「你怎么理解 messages 结构 / tool call 时序 / 流式输出」，答案就藏在你亲手写这三份脚本的过程里。

先读完 `src/agent_core/llm.py`（有中文注释，逐行看懂），再动手。

---

## 练习 1：`playground/01_text.py` —— 非流式文本调用

**要求**
1. 用 `LLMClient().chat(...)` 问一个需要一点推理的问题（比如「用三句话解释什么是 Agent 的上下文工程」）。
2. 打印：回答正文、`model`、`prompt_tokens` / `completion_tokens` / `total_tokens`、`latency_ms`。
3. **再调用一次同样的 questions，但把 `temperature` 设为 0.0 和 1.2 各跑一次**，观察输出差异，把结论写在文件末尾的注释里（3 行以内）。

**验收**
- [ ] 终端能打印出完整的回答和三项 token 数
- [ ] 你能回答：为什么 `messages[0]` 是 `system`，为什么最后一条必须是 `user`

---

## 练习 2：`playground/02_stream.py` —— 流式调用

**要求**
1. 用 `LLMClient().stream(...)` 让回答逐字打印到终端（用 `end=""` + `flush=True`）。
2. 自己记录 **首字延迟（TTFT）** 和 **总耗时**，结束后打印出来。
3. 从最后一个分片里取 `usage` 并打印。

**验收**
- [ ] 肉眼看得出是逐字出来的，不是一次性打印
- [ ] 你打印出了 TTFT 和总耗时两个数字
- [ ] 你能解释：**为什么流式下 usage 只在最后一个分片里出现**，以及为什么 `finish_reason="stop"` 的分片 `delta` 通常为空

> 这两个延迟指标（TTFT / 总耗时）在 D18 接上 Langfuse 后会成为你项目的正式指标。
> 今天先手工测，是为了让你知道这些数字是从哪里来的。

---

## 练习 3：`playground/03_vision.py` —— 图片理解

**要求**
1. 准备一张图（项目截图、手写笔记、或者一页文档的照片都行），放到 `playground/sample.png`。
2. 用 `LLMClient().chat_with_image(...)` 让模型描述图片内容，或者做一次 OCR。
3. 打印回答 + token 用量。

**验收**
- [ ] 模型能正确描述你给的那张图
- [ ] 你能解释：图片是怎么变成 input token 的（data URL / base64 / 多模态 content 数组）

> 这一步是 D4「文档摄取」的伏笔——扫描件和截图都要先走视觉模型变成文本，才能进 RAG。

---

## 练习 4：`playground/04_token_counter.py` —— token 计数器

**要求**
写一个可复用的函数，而不是一次性脚本：

```python
def count_and_cost(text_or_messages, model: str) -> dict:
    """返回 {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int, "cost_cny": float | None}"""
```

实现方式二选一（都做更好）：
- **方式 A（准确，需网络）**：真调一次 API，从 `usage` 里读真实 token 数。
- **方式 B（离线估算）**：用一个 tokenizer 估算，不花 token。可选 `tiktoken`，或者中英文分开粗算
  （经验值：中文约 1 字 ≈ 0.6-1.5 token，英文约 1 词 ≈ 1.3 token，代码里符号各占 token）。

**然后在 `agent_core/config.py` 的 `PRICES_CNY_PER_1K` 里填入你在控制台查到的真实价格**，
让 `estimate_cost_cny()` 能算出钱。

**验收**
- [ ] 能对一段中英混排 + 一段代码分别报出 token 数和估算费用
- [ ] 你把「方式 A 和方式 B 的误差有多大」写进了注释（这是很值钱的工程细节，面试可讲）

> **为什么今天就要做这个**：JD 里反复出现「模型调用成本」「Token 消耗」。
> 到 D17 你要给出「单条任务成本从 X 元降到 Y 元」的数字——没有这个计数器，那个数字就是编的。

---

## 完成后

```bash
# 全部跑一遍，确认没问题
uv run python playground/01_text.py
uv run python playground/02_stream.py
uv run python playground/03_vision.py
uv run python playground/04_token_counter.py

# 质量检查（应该全绿）
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q

# 提交
git add -A && git commit -m "D1: 环境起盘 + 统一模型接入层 + 三个最小调用 + token 计数器"
```

最后把 D1 的打卡行填了（`AI全栈转型/13-4周极速冲刺计划（国内线）.md` 第五节 Week 1）。
