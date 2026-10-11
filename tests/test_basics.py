"""不依赖网络的单测：保证 CI 在没有密钥的环境里也能跑绿。

真正调 API 的测试放到 tests/integration/ 下，并用 pytest mark 标记，
默认不跑（后面几天会加）。

跑法：  uv run pytest -q
"""

from __future__ import annotations

import json

from agent_core.config import Settings
from agent_core.llm import Usage, _extract_tool_calls, _usage_from, estimate_cost_cny, system_user


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, cid: str, name: str, arguments: str) -> None:
        self.id = cid
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall] | None = None) -> None:
        if tool_calls is not None:
            self.tool_calls = tool_calls


def test_extract_tool_calls_empty() -> None:
    """没有 tool_calls 字段、或字段为空时都要返回空列表，不能抛异常。"""
    assert _extract_tool_calls(_FakeMessage()) == []
    assert _extract_tool_calls(_FakeMessage([])) == []


def test_extract_tool_calls_shape() -> None:
    """回归用例：tool calling 时参数在 arguments 里，而且它是字符串。

    实测走 tool calling 时 content 是空字符串，参数全在这里。
    不提取它，D3 的 Agent 循环就拿不到模型想干什么。
    """
    msg = _FakeMessage([_FakeToolCall("call_1", "get_weather", '{"city":"北京"}')])
    calls = _extract_tool_calls(msg)
    assert len(calls) == 1
    assert calls[0]["id"] == "call_1"
    assert calls[0]["name"] == "get_weather"
    # 刻意不做 json.loads：解析失败也要让上层看得见，不能提前吞掉
    assert isinstance(calls[0]["arguments"], str)
    assert json.loads(calls[0]["arguments"]) == {"city": "北京"}


class _FakeDetails:
    def __init__(self, reasoning_tokens: int) -> None:
        self.reasoning_tokens = reasoning_tokens


class _FakeUsage:
    """模拟 SDK 的 usage 对象（含推理模型才会有的 completion_tokens_details）。"""

    def __init__(self, p: int, c: int, t: int, reasoning: int | None = None) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t
        if reasoning is not None:
            self.completion_tokens_details = _FakeDetails(reasoning)


def test_usage_addition() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert (a + b) == Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18)


def test_usage_addition_carries_reasoning_tokens() -> None:
    """回归用例：加总时必须带上 reasoning_tokens。

    漏掉这个字段会让多次调用的思维链成本悄悄消失——
    而实测思维链占输出成本的 36%，不见了就是一笔糊涂账。
    """
    a = Usage(10, 20, 30, reasoning_tokens=12)
    b = Usage(1, 2, 3, reasoning_tokens=1)
    total = a + b
    assert total.reasoning_tokens == 13
    assert total.visible_tokens == 22 - 13


def test_visible_tokens() -> None:
    assert Usage(10, 100, 110, reasoning_tokens=60).visible_tokens == 40
    # provider 不返回 reasoning 明细时，visible == completion，不能算成负数
    assert Usage(10, 100, 110).visible_tokens == 100
    # 脏数据保护：reasoning 比 completion 还大时不返回负数
    assert Usage(10, 10, 20, reasoning_tokens=99).visible_tokens == 0


def test_usage_from_handles_missing_details() -> None:
    """不是所有 provider 都返回 completion_tokens_details，取不到就当 0。"""
    assert _usage_from(None) == Usage()
    assert _usage_from(_FakeUsage(5, 6, 11)).reasoning_tokens == 0
    with_details = _usage_from(_FakeUsage(5, 6, 11, reasoning=4))
    assert with_details.reasoning_tokens == 4
    assert with_details.visible_tokens == 2


def test_system_user_shape() -> None:
    msgs = system_user("你是助手", "你好")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[-1]["role"] == "user", "最后一条必须是 user，否则多数模型会报错"


def test_cost_is_none_for_unpriced_model() -> None:
    """没登记价格的模型必须返回 None，而不是编一个数字出来。"""
    assert estimate_cost_cny(Usage(1, 1, 2), "不存在的模型") is None


def test_settings_defaults() -> None:
    """Settings 应该能在没有 .env 的情况下构造出来（全部有默认值）。"""
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider in {"dashscope", "ark"}
    assert s.dashscope_base_url.startswith("https://")
    assert s.ark_base_url.startswith("https://")


def test_missing_key_raises_helpful_error() -> None:
    """没配 key 时应该报一条能指导操作的错，而不是底层 SDK 的 401。"""
    from agent_core.llm import LLMClient

    s = Settings(_env_file=None)  # type: ignore[call-arg]
    s.dashscope_api_key = ""
    try:
        LLMClient(settings=s).chat(system_user("s", "u"))
    except RuntimeError as exc:
        assert "API Key" in str(exc)
    else:
        raise AssertionError("没配 key 时应该抛 RuntimeError")
