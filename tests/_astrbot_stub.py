"""测试环境 astrbot 替身：仅在未安装 astrbot 时注入最小 logger。

store.py 按市场审核规范必须 `from astrbot.api import logger`。本地单元测试
环境通常没有完整 astrbot 运行时，这里在导入插件包之前先注册一个只含
logger 的假 astrbot.api 模块，保证测试可独立运行；真实 astrbot 已安装时
本模块不做任何事。不使用 Python 内置 logging，避免触发审核扫描。
"""
from __future__ import annotations

import sys
import types


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


try:  # pragma: no cover - 取决于本地是否安装 astrbot
    import astrbot.api  # noqa: F401
except ImportError:
    _astrbot = types.ModuleType("astrbot")
    _api = types.ModuleType("astrbot.api")
    _api.logger = _NoopLogger()
    _astrbot.api = _api
    sys.modules.setdefault("astrbot", _astrbot)
    sys.modules.setdefault("astrbot.api", _api)
