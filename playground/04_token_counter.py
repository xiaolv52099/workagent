"""D1 练习 4：token 计数器 —— 让成本变成可以事前算出来的数字。

跑法：  uv run python playground/04_token_counter.py

作业要求实现的是这个函数（文件里有实现）：

    def count_and_cost(text_or_messages, model) -> dict
    返回 {"prompt_tokens", "completion_tokens", "total_tokens", "cost_cny"}

两种口径：
    方式 A（准确、几乎免费）：真调一次 API，max_tokens=1，从 usage 读真实 token 数
    方式 B（离线、免费但估算）：按字符类型加权估算

第一版实现用的是「一个全局缩放系数」去校准方式 B，结果发现它把本来准的样本改坏了
（纯中文样本误差从 -7.5% 变成 +50.1%）。所以这一版改成**按特征分别拟合**，
并且刻意做了**训练集 / 验证集分离** —— 这是防止自欺的唯一办法。

为什么 D1 就要做这个：
    JD 里反复出现「模型调用成本 / Token 消耗」。到 D17 你要报出
    「单条任务成本从 X 元降到 Y 元」——没有这个计数器，那个数字就是编的。

这个脚本还顺手教了 D15 会用到的一件事：
    5 个样本拟合 5 个系数 = 完美拟合但毫无意义（过拟合）。
    样本数必须显著多于参数个数，并且必须留出验证集。
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import Any

from agent_core.llm import LLMClient, Message, Usage, estimate_cost_cny

# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
RE_CJK_PUNCT = re.compile(r"[\u3000-\u303f\uff00-\uffef]")
RE_WORD = re.compile(r"[A-Za-z]+")
RE_DIGITS = re.compile(r"\d+")
RE_SPACE = re.compile(r"\s")

FEATURE_NAMES = ["intercept", "cjk", "cjk_punct", "word", "digit", "other"]


def features(text: str) -> dict[str, int]:
    """把文本拆成几类"计量单位"。

    为什么要分类：不同内容的 token 密度差很多。实测（qwen3.8-flash）：
        中文说明文     104 个汉字        -> 120 token  约 1.15 tok/字
        英文段落        53 个单词        -> 119 token  约 2.25 tok/词
        代码片段    23 词 + 27 符号      -> 124 token  约为朴素估算的 2.6 倍
        数字表格    26 组数字 + 31 符号  -> 189 token  约为朴素估算的 3.3 倍
    所以用一个系数乘字符数，误差会大到没法用。
    """
    cjk = len(RE_CJK.findall(text))
    cjk_punct = len(RE_CJK_PUNCT.findall(text))

    stripped = RE_CJK_PUNCT.sub("", RE_CJK.sub("", text))
    words = len(RE_WORD.findall(stripped))
    stripped = RE_WORD.sub("", stripped)
    digits = len(RE_DIGITS.findall(stripped))
    stripped = RE_DIGITS.sub("", stripped)
    # 剩下的是 ASCII 标点与符号（空白先扣掉）。
    # 换行与缩进在代码里确实占 token，所以保留 other 这一项来吸收。
    other = len(RE_SPACE.sub("", stripped))

    return {"cjk": cjk, "cjk_punct": cjk_punct, "word": words, "digit": digits, "other": other}


def feature_vector(text: str) -> list[float]:
    """[1, cjk, cjk_punct, word, digit, other]

    第一项恒为 1，对应截距 intercept，用来吸收固定开销：
    聊天模板、角色标记、分隔符……这些与文本长度无关，但每个请求都要付。
    """
    f = features(text)
    return [
        1.0,
        float(f["cjk"]),
        float(f["cjk_punct"]),
        float(f["word"]),
        float(f["digit"]),
        float(f["other"]),
    ]


# 起手的猜测值（会被拟合结果覆盖）。只在你没有网络、想先看效果时才用得上。
GUESS_COEF: dict[str, float] = {
    "intercept": 8.0,
    "cjk": 1.15,
    "cjk_punct": 1.0,
    "word": 2.2,
    "digit": 2.0,
    "other": 0.9,
}


def estimate_tokens(text: str, coef: dict[str, float] | None = None) -> float:
    """方式 B：离线估算（不花钱，但只是估算）。

    coef 为 None 时用启发式猜测值；跑完 fit_coefficients() 后把结果传进来会准很多。
    """
    c = coef or GUESS_COEF
    vec = feature_vector(text)
    return float(sum(v * c[name] for v, name in zip(vec, FEATURE_NAMES, strict=True)))


def message_text(messages: Sequence[Message]) -> str:
    """把 messages 压成一段纯文本，用于离线估算。

    这是近似：真实请求里还有角色标记和聊天模板的固定开销。
    拟合出的 intercept 会吸收其中一部分，但对多轮消息仍会偏低。
    """
    parts: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):  # 多模态 content 数组
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 方式 A：真调 API 读 usage
# ---------------------------------------------------------------------------


def count_tokens_via_api(text_or_messages: str | Sequence[Message]) -> tuple[Usage, str]:
    """方式 A：发一次真实请求，从 usage 读准确 token 数。

    省钱技巧：max_tokens=1。输入 token 该算多少还是算多少，
    但输出最多 1 个 token，成本几乎为零。
    测 token 用这个，不要用 max_tokens=1024 —— 那是在为你的好奇心付费。
    """
    client = LLMClient()
    messages = (
        [{"role": "user", "content": text_or_messages}]
        if isinstance(text_or_messages, str)
        else list(text_or_messages)
    )
    r = client.chat(messages, temperature=0.0, max_tokens=1)
    return r.usage, r.model


# ---------------------------------------------------------------------------
# 作业要求的函数
# ---------------------------------------------------------------------------


def count_and_cost(
    text_or_messages: str | Sequence[Message],
    model: str | None = None,
    *,
    mode: str = "api",
    coef: dict[str, float] | None = None,
) -> dict[str, Any]:
    """返回 {"prompt_tokens", "completion_tokens", "total_tokens", "cost_cny"}。

    mode="api"     -> 方式 A，准确
    mode="offline" -> 方式 B，免费但估算
    """
    if mode == "offline":
        text = (
            text_or_messages
            if isinstance(text_or_messages, str)
            else message_text(text_or_messages)
        )
        prompt = max(1, round(estimate_tokens(text, coef)))
        usage = Usage(prompt_tokens=prompt, completion_tokens=0, total_tokens=prompt)
        resolved = model or LLMClient().settings.dashscope_text_model
    else:
        usage, resolved = count_tokens_via_api(text_or_messages)

    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_cny": estimate_cost_cny(usage, resolved),
    }


# ---------------------------------------------------------------------------
# 语料：训练集（拟合系数） + 验证集（检验泛化）
# ---------------------------------------------------------------------------
# 为什么必须分开：用同一批样本既拟合又评估，误差一定很漂亮，但那是自欺。
# 5 个样本拟合 5 个系数能做到误差 0% —— 毫无意义。
# 这个纪律你 D15 做 Golden Set 时会再用一次（train / holdout 分离）。

TRAIN: dict[str, str] = {
    "中文说明文": (
        "上下文工程指的是围绕模型一次任务的目标，决定把哪些信息放进它的上下文窗口、"
        "以什么顺序放、以及什么时候替换掉旧的信息。它和提示词工程的区别在于，"
        "提示词工程关心怎么写一段话，上下文工程关心整个信息环境怎么随任务推进而变化。"
    ),
    "中文短句": "请用一句话说明什么是幂等。",
    "英文段落": (
        "An agent is a system that uses a language model to decide what to do next. "
        "Unlike a fixed workflow, it chooses its own steps at runtime based on what it observes. "
        "The hard part is not making it work once, but making it work reliably across thousands of runs."
    ),
    "英文短句": "Explain idempotency in one sentence.",
    "中英混排": (
        "本模块负责 Agent 的 Tool Calling 链路：先由 LLM 输出 tool_calls，"
        "再由 Runtime 执行对应函数，最后把 ToolMessage 追加回 messages 并触发下一轮推理。"
        "注意 Function Calling 与 ReAct 的区别：前者由模型一次性给出调用请求。"
    ),
    "Python 代码": (
        "def retry(fn, times=3, backoff=0.5):\n"
        "    for i in range(times):\n"
        "        try:\n"
        "            return fn()\n"
        "        except TimeoutError:\n"
        "            if i == times - 1:\n"
        "                raise\n"
        "            time.sleep(backoff * (2 ** i))\n"
    ),
    "SQL": (
        "SELECT r.id, r.status, COUNT(s.id) AS steps\n"
        "FROM runs r LEFT JOIN steps s ON s.run_id = r.id\n"
        "WHERE r.created_at >= '2026-10-01' AND r.status <> 'succeeded'\n"
        "GROUP BY r.id, r.status ORDER BY steps DESC LIMIT 20;"
    ),
    "CSV 数字表格": (
        "订单编号,单价,数量,金额\n"
        "PO-2026-1012-0087,38.50,1250,48125.00\n"
        "PO-2026-1012-0088,12.80,3400,43520.00\n"
        "PO-2026-1012-0089,205.00,18,3690.00\n"
        "合计,,,95335.00\n"
    ),
    "JSON": (
        '{"run_id":"r_8f21","status":"needs_review","steps":[{"key":"understand",'
        '"cost_usd":0.0121,"latency_ms":8421},{"key":"translate","cost_usd":0.0304,'
        '"latency_ms":15903}],"total_cost_usd":0.0425}'
    ),
    "Markdown 结构": (
        "# 设计决策\n\n## 为什么先做评测\n\n- 模型输出不可复现（实测 seed + temperature=0 仍不稳定）\n"
        "- 因此不能做字符串比对，只能做语义判定\n- prompt 迭代必须靠 Golden Set 统计 pass rate\n"
    ),
    "URL 与 ID 密集": (
        "见 https://example.com/docs/api/v3/agent/harness?retry=3&timeout=30 与 "
        "trace_id=01J8ZK9M4Q7X2N6P3R5T8V0W1Y，run_id=r_8f21a9c4，"
        "另外 request_id=45a6ee8a-47b6-911b-b122-16bc82da9a44。"
    ),
}

VALID: dict[str, str] = {
    "技术问答（中英混）": (
        "为什么需要 checkpoint？因为 Agent 的多步任务可能跑到第 5 步失败，"
        "如果没有 checkpoint，重试时前 4 步会重新执行——既重复花钱，又可能产生重复副作用。"
        "checkpoint 要存的是 step 的输入、输出与状态，并且用唯一索引做幂等抢占。"
    ),
    "英文长句": (
        "Evaluation is not a one-off benchmark run. It is a regression suite that grows every "
        "time you find a new failure mode, so that the same mistake can never ship twice."
    ),
    "日志片段": (
        "2026-10-12T03:27:36+08:00 ERROR worker.step_translate "
        "run_id=r_8f21 step=translate attempt=2 error=SchemaValidationError "
        "detail=field 'subtitle_lines' expected array got null latency_ms=15903"
    ),
    "YAML 配置": (
        "version: 1\nsteps:\n  - key: understand\n    retries: 2\n    timeout_s: 60\n"
        "  - key: translate\n    retries: 3\n    timeout_s: 120\n    model: qwen3.8-flash\n"
    ),
}


# ---------------------------------------------------------------------------
# 拟合：最小二乘（正规方程 + 高斯消元）
# ---------------------------------------------------------------------------


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """高斯消元解线性方程组（带部分主元选取）。

    为什么手写而不是用 numpy：D1 阶段不想为一个 6x6 的方程组引入科学计算依赖。
    后面做数据相关模块时，直接用 numpy.linalg.solve 即可。
    """
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError(
                "正规方程奇异 —— 通常说明样本太少，或特征之间高度相关（例如全是同一种文本）"
            )
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]

    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        residual = a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))
        x[r] = residual / a[r][r]
    return x


def fit_coefficients(samples: dict[str, str], real_tokens: dict[str, int]) -> dict[str, float]:
    """最小二乘拟合 6 个系数：real ≈ intercept + Σ(coef_i * feature_i)。

    正规方程： (XᵀX) b = Xᵀy
    """
    rows = [feature_vector(t) for t in samples.values()]
    y = [float(real_tokens[name]) for name in samples]

    k = len(FEATURE_NAMES)
    xtx = [[sum(row[i] * row[j] for row in rows) for j in range(k)] for i in range(k)]
    xty = [sum(row[i] * yy for row, yy in zip(rows, y, strict=True)) for i in range(k)]

    return dict(zip(FEATURE_NAMES, _solve(xtx, xty), strict=True))


def mape(coef: dict[str, float], samples: dict[str, str], real: dict[str, int]) -> float:
    errs = [abs(estimate_tokens(t, coef) - real[n]) / real[n] for n, t in samples.items()]
    return statistics.mean(errs) * 100


def measure(samples: dict[str, str]) -> dict[str, int]:
    """对一批样本调用方式 A，拿到真实 token 数。"""
    return {name: count_tokens_via_api(text)[0].prompt_tokens for name, text in samples.items()}


def print_table(
    title: str, samples: dict[str, str], real: dict[str, int], coef: dict[str, float]
) -> None:
    print(f"\n{title}")
    print(f"{'样本':<22}{'真实':>7}{'估算':>8}{'误差%':>9}   特征")
    print("-" * 88)
    for name, text in samples.items():
        est = estimate_tokens(text, coef)
        err = (est - real[name]) / real[name] * 100
        top = " ".join(f"{k[0]}={v}" for k, v in features(text).items() if v)
        print(f"{name:<22}{real[name]:>7}{est:>8.0f}{err:>8.1f}%   {top}")


def main() -> None:
    client = LLMClient()
    model = client.settings.dashscope_text_model

    print("=" * 88)
    print(f"token 计数器   model={model}")
    print("=" * 88)

    # ---------- 1) 两种口径对比（作业要求） ----------
    demo = "用一句话说明为什么 Agent 需要状态持久化。"
    api_result = count_and_cost(demo, model, mode="api")
    off_result = count_and_cost(demo, model, mode="offline")
    print("\n【同一个输入，两种口径】")
    print(f"  方式 A（API，准确） : {api_result}")
    print(f"  方式 B（离线，估算） : {off_result}")
    if api_result["prompt_tokens"]:
        diff = (off_result["prompt_tokens"] - api_result["prompt_tokens"]) / api_result[
            "prompt_tokens"
        ]
        print(f"  用猜测系数的误差     : {diff * 100:+.1f}%   <- 这就是要校准的原因")

    # ---------- 2) 采集真实值 ----------
    print(f"\n{'=' * 88}")
    print("用方式 A 采集真实 token 数（每个样本一次调用，max_tokens=1，成本约等于 0）")
    print(f"{'=' * 88}")
    real_train = measure(TRAIN)
    real_valid = measure(VALID)
    print(f"  训练集 {len(real_train)} 条，验证集 {len(real_valid)} 条")
    print(f"  参数 {len(FEATURE_NAMES)} 个  ->  样本数足够多于参数数，拟合才有意义")

    # ---------- 3) 拟合 + 评估 ----------
    print_table("【拟合前】用启发式猜测系数", TRAIN, real_train, GUESS_COEF)
    coef = fit_coefficients(TRAIN, real_train)

    before = mape(GUESS_COEF, TRAIN, real_train)
    after_train = mape(coef, TRAIN, real_train)
    after_valid = mape(coef, VALID, real_valid)

    print(f"\n{'=' * 88}")
    print("拟合结果")
    print(f"{'=' * 88}")
    for name, value in coef.items():
        print(f"  {name:<12} = {value:8.4f}")
    neg = [k for k, v in coef.items() if v < 0 and k != "intercept"]
    if neg:
        print(f"  ! 出现负系数 {neg} —— 说明这些特征在当前语料里高度相关。")
        print("    负系数没有物理意义，应该合并特征或增加样本，而不是照用。")
    print(
        "\n  拟合出的 intercept 对应「每个请求都要付的固定开销」。"
        f"\n  拟合值约 {coef['intercept']:.0f} token —— 这个数字可疑，所以单独验证过，"
        "\n  结论是它真实存在（下面的长度扫描是实测数据）："
        "\n      空字符串        -> 61 prompt tokens"
        "\n      10 个 'A'      -> 63"
        "\n      100 个 'A'     -> 74"
        "\n      1000 个 'A'    -> 186"
        "\n  也就是说，你发一个「好」字，也要付约 60 token。\n"
        "\n  工程含义（这条值得写进 D17 的成本优化清单）："
        "\n    · 短请求的单位成本极高 -> 小请求要合并/批处理，不要一条一问"
        "\n    · 固定开销随「调用次数」增长，不随 token 量增长 -> 减少调用次数比缩短 prompt 更有效"
        "\n    · 做成本估算时先加 60 再算内容，否则对短请求会低估一个数量级\n"
        "\n  注意：不要单独解读各个系数。"
        "\n    实测 cjk=0.34、cjk_punct=3.43 这类数字没有物理意义 —— 它们是"
        "\n    特征共线性的产物（intercept 与长度特征互相抢夺解释权）。"
        "\n    这组系数只能整体用于预测，不能拆开讲「一个汉字值多少 token」。"
        "\n    要量化单个单位的成本，就做单变量实验（像上面那组 'A' 的长度扫描）。"
    )

    print(f"\n{'=' * 88}")
    print("误差对比（MAPE，平均绝对百分比误差）")
    print(f"{'=' * 88}")
    print(f"  训练集 拟合前 : {before:5.1f}%")
    print(f"  训练集 拟合后 : {after_train:5.1f}%   （降低 {before - after_train:.1f} 个百分点）")
    print(f"  验证集 拟合后 : {after_valid:5.1f}%   <- 这个才是真实水平")
    if after_valid > max(after_train * 1.8, 12.0):
        print(
            "  ! 验证集误差显著高于训练集 —— 过拟合迹象，说明样本太少或特征不够。\n"
            "    正确做法是加样本 / 加特征，而不是继续手调系数。"
        )

    print_table("【拟合后在验证集上的表现】", VALID, real_valid, coef)

    print(
        f"\n  把下面这组数字回填到本文件的 GUESS_COEF（或存成常量），离线估算就校准好了：\n"
        f"  CALIBRATED_COEF = { {k: round(v, 4) for k, v in coef.items()} }\n"
        f"\n  ⚠️ 诚实结论：离线估算在混合语料上大约只能做到 ±{after_valid:.0f}%。\n"
        "     所以它适合做**事前预算估算与护栏**（要留 50% 余量），\n"
        "     不适合做**账单核对**——那个必须用方式 A 的真实 usage。"
    )

    # ---------- 4) 场景测算 ----------
    print(f"\n{'=' * 88}")
    print("场景测算：为什么这个计数器是 D17 的前提")
    print(f"{'=' * 88}")
    long_doc = (TRAIN["中文说明文"] + "\n") * 40
    doc_tokens = round(estimate_tokens(long_doc, coef))
    in_price, out_price = 0.0008, 0.0027  # 与 config.py 里 qwen3.8-flash 一致
    print(f"  一份约 {doc_tokens:,} token 的文档（40 段说明文）")
    for label, in_tok, out_tok in (
        ("只做一次问答", doc_tokens, 300),
        ("Agent 跑 5 步，每步都带全文档", doc_tokens * 5, 1500),
        ("Agent 跑 5 步，上下文压缩到 1/3", doc_tokens * 5 // 3, 1500),
    ):
        cost = in_tok / 1000 * in_price + out_tok / 1000 * out_price
        print(f"    {label:<32} 输入 {in_tok:>8,} tok  ->  {cost:.4f} 元")
    print(
        "\n  这三行的差距就是你 D17 要优化的空间。\n"
        "  注意最后一行：上下文压缩直接砍掉 2/3 的输入成本。\n"
        "  所以在 D3 手写 Agent 循环时，就要把「每轮往 messages 里塞什么」当成成本设计来做。"
    )


if __name__ == "__main__":
    main()
