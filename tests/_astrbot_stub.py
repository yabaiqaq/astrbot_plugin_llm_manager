"""测试环境 astrbot 替身：仅在未安装 astrbot 时注入最小运行时。

store.py 按市场审核规范必须 `from astrbot.api import logger`。本地单元测试
环境通常没有完整 astrbot 运行时，这里在导入插件包之前先注册最小模块，
保证测试可独立运行；真实 astrbot 已安装时本模块不做任何事。
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path


class _NoopLogger:
    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def critical(self, *args, **kwargs) -> None:
        pass


class _Provider:
    """manager_provider 所需的最小 Provider 基类。"""

    def __init__(self, provider_config=None, provider_settings=None) -> None:
        self.provider_config = dict(provider_config or {})
        self.provider_settings = provider_settings or {}
        self.model = str(self.provider_config.get("model") or "")

    def set_model(self, model: str) -> None:
        self.model = str(model)

    def get_model(self) -> str:
        return self.model

    async def test(self) -> bool:
        return True


class _StarTools:
    @staticmethod
    def get_data_dir(name: str) -> Path:
        return Path(tempfile.gettempdir()) / "astrbot_test_data" / name


def _install_stub() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _NoopLogger()

    provider_api = types.ModuleType("astrbot.api.provider")
    provider_api.Provider = _Provider
    api.provider = provider_api

    star_api = types.ModuleType("astrbot.api.star")
    star_api.StarTools = _StarTools
    api.star = star_api

    core = types.ModuleType("astrbot.core")
    core_provider = types.ModuleType("astrbot.core.provider")
    register = types.ModuleType("astrbot.core.provider.register")
    register.provider_cls_map = {}
    register.provider_registry = []
    core_provider.register = register
    core.provider = core_provider
    astrbot.api = api
    astrbot.core = core

    for name, module in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.provider": provider_api,
        "astrbot.api.star": star_api,
        "astrbot.core": core,
        "astrbot.core.provider": core_provider,
        "astrbot.core.provider.register": register,
    }.items():
        sys.modules.setdefault(name, module)


try:  # pragma: no cover - 取决于本地是否安装 astrbot
    import astrbot.api  # noqa: F401
except ImportError:
    _install_stub()
