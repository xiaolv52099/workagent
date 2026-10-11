"""配置层：所有配置与密钥只从环境变量 / .env 读取。

为什么单独一个文件：
1. 密钥不进代码（constitution 里的安全红线）；
2. 换模型、换 provider、调超时都只改这里；
3. 测试时可以用 Settings(...) 覆盖，不用改环境。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["dashscope", "ark"]


class Settings(BaseSettings):
    """字段名会自动匹配同名的大写环境变量（大小写不敏感）。

    例如 dashscope_api_key  ←→  DASHSCOPE_API_KEY
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有额外变量不报错
    )

    # --- 运行参数 ---
    llm_provider: Provider = "dashscope"
    request_timeout_seconds: float = 60.0
    max_retries: int = 2

    # --- 阿里云百炼（DashScope / 千问）---
    dashscope_api_key: str = ""
    # 老域名不需要 WorkspaceId，D1 先用它；跑通后再迁移到官方推荐的业务空间专属域名
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_text_model: str = "qwen-plus"
    dashscope_vision_model: str = "qwen-vl-max"

    # --- 火山方舟（Ark / 豆包）---
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3/"
    ark_text_model: str = ""
    ark_vision_model: str = ""

    def api_key_for(self, provider: Provider) -> str:
        return self.dashscope_api_key if provider == "dashscope" else self.ark_api_key

    def base_url_for(self, provider: Provider) -> str:
        return self.dashscope_base_url if provider == "dashscope" else self.ark_base_url

    def model_for(self, provider: Provider, *, vision: bool = False) -> str:
        if provider == "dashscope":
            return self.dashscope_vision_model if vision else self.dashscope_text_model
        return self.ark_vision_model if vision else self.ark_text_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内缓存的单例。改 .env 后需要重启进程才生效。"""
    return Settings()
