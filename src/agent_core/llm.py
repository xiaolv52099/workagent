"""统一模型接入层（Unified LLM Gateway）。

为什么需要这一层（面试也会问）：
    职位描述里反复出现「多模型路由 / 模型降级 / Token 成本控制 / 超时重试」。
    如果业务代码里到处写 OpenAI(base_url=..., api_key=...)，换模型就要改 N 个地方，
    也没法在一个地方统一记成本、统一加重试、统一做降级链。
    所以把 provider 差异全部收敛到这里，业务层只认 chat() / stream()。

本文件在 D1 只做三件事：非流式调用、流式调用、视觉调用 + 用量记账。
D16-D18 会在这层之上加：模型降级链、预算护栏、Langfuse trace。
"""

from __future__ import annotations

import base64
import mimetypes
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import Provider, Settings, get_settings

# ---------------------------------------------------------------------------
# 用量与成本
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """一次调用的 token 用量。OpenAI 兼容协议的 usage 字段就是这三个数。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


# TODO(D1): 把下面的单价换成你在控制台查到的真实价格（单位：元 / 千 token）。
# 价格会调整，所以这里不写死猜测值，查不到就返回 None，不伪造数字。
PRICES_CNY_PER_1K: dict[str, tuple[float, float]] = {
    # "模型名": (输入价, 输出价)
}


def estimate_cost_cny(usage: Usage, model: str) -> float | None:
    """估算成本。未登记价格的模型返回 None（诚实优先于给个假数）。"""
    price = PRICES_CNY_PER_1K.get(model)
    if price is None:
        return None
    in_price, out_price = price
    return usage.prompt_tokens / 1000 * in_price + usage.completion_tokens / 1000 * out_price


# ---------------------------------------------------------------------------
# 返回结构
# ---------------------------------------------------------------------------


@dataclass
class ChatResult:
    text: str
    model: str
    usage: Usage
    latency_ms: float
    finish_reason: str | None = None


@dataclass
class StreamChunk:
    """流式响应的一个分片。

    注意：只有在 stream_options={"include_usage": True} 时，
    usage 才会出现在最后一个分片里（此时该分片 choices 为空）。
    """

    delta: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

Message = dict[str, Any]


def _to_sdk_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    """把我们宽松的 Message 转成 SDK 的严格类型。

    为什么需要这一层：我们会在运行时动态拼消息（加 tool 结果、加图片块、加记忆），
    用 dict 最方便；而 SDK 用的是很窄的 TypedDict 联合类型。
    与其在每个调用点写 `# type: ignore`，不如在这里 cast 一次并说明原因——
    类型忽略要集中、要有理由，散落各处就没人敢改了。
    """
    return cast("list[ChatCompletionMessageParam]", list(messages))


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._clients: dict[Provider, OpenAI] = {}

    # --- 内部：懒加载 provider 客户端 ---
    def _client(self, provider: Provider) -> OpenAI:
        if provider in self._clients:
            return self._clients[provider]

        api_key = self.settings.api_key_for(provider)
        base_url = self.settings.base_url_for(provider)
        if not api_key or api_key.startswith(("sk-xxx", "xxxx")):
            raise RuntimeError(
                f"{provider} 的 API Key 没配。请 cp .env.example .env 并填好 "
                f"{provider.upper()}_API_KEY（或对应的 *_API_KEY 变量）。"
            )

        self._clients[provider] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.settings.request_timeout_seconds,
            max_retries=self.settings.max_retries,
        )
        return self._clients[provider]

    def _resolve(self, provider: Provider | None, *, vision: bool) -> tuple[OpenAI, str, Provider]:
        name: Provider = provider or self.settings.llm_provider
        model = self.settings.model_for(name, vision=vision)
        if not model:
            raise RuntimeError(
                f"{name} 的{'视觉' if vision else '文本'}模型名没配。"
                "请到控制台「模型广场 / 开通管理」复制当前可用模型名填进 .env。"
            )
        return self._client(name), model, name

    # ------------------------------------------------------------------
    # 1) 非流式
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        provider: Provider | None = None,
        **extra: Any,
    ) -> ChatResult:
        client, model, _ = self._resolve(provider, vision=False)
        started = time.perf_counter()

        resp = client.chat.completions.create(
            model=model,
            messages=_to_sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        choice = resp.choices[0]
        u = resp.usage
        return ChatResult(
            text=choice.message.content or "",
            model=resp.model or model,
            usage=Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
        )

    # ------------------------------------------------------------------
    # 2) 流式
    # ------------------------------------------------------------------
    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        provider: Provider | None = None,
        include_usage: bool = True,
        **extra: Any,
    ) -> Iterator[StreamChunk]:
        """逐字返回。

        include_usage=True 会在最后一个分片带上 usage。
        如果某个 provider 报错说 stream_options 不支持，把它设成 False 即可。
        """
        client, model, _ = self._resolve(provider, vision=False)
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=_to_sdk_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **extra,
        )
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        # 注意：stream=True 时返回值是可迭代的 Stream；mypy 能通过 openai 的重载推断出来，
        # 所以这里不需要 type: ignore。如果哪天升级 SDK 后报类型错误，先看返回类型再决定加不加。
        for chunk in client.chat.completions.create(**kwargs):
            delta = ""
            finish_reason: str | None = None
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
                finish_reason = chunk.choices[0].finish_reason

            usage_obj = getattr(chunk, "usage", None)
            usage = None
            if usage_obj is not None:
                usage = Usage(
                    prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
                )
            yield StreamChunk(delta=delta, usage=usage, finish_reason=finish_reason)

    # ------------------------------------------------------------------
    # 3) 视觉（图片输入）
    # ------------------------------------------------------------------
    @staticmethod
    def image_to_data_url(image_path: str | Path) -> str:
        """把本地图片转成 data URL（base64），这样不用先传对象存储。"""
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def chat_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        provider: Provider | None = None,
        temperature: float = 0.2,
    ) -> ChatResult:
        client, model, _ = self._resolve(provider, vision=True)
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self.image_to_data_url(image_path)}},
                ],
            }
        ]
        started = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=_to_sdk_messages(messages),
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        choice = resp.choices[0]
        u = resp.usage
        return ChatResult(
            text=choice.message.content or "",
            model=resp.model or model,
            usage=Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            ),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason,
        )


def system_user(system: str, user: str) -> list[Message]:
    """最常用的消息结构，省得每次手写 dict。"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
