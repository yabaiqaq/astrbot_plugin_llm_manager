"""运行时依赖检测与自动安装。

在插件模块加载阶段调用 ensure_pillow()，若 Pillow 缺失则自动 pip 安装，
确保后续 renderer.py 的 `from PIL import ...` 能成功。
"""
from __future__ import annotations

import importlib
import subprocess
import sys

from astrbot import logger

_PILLOW_MIN_VERSION = "10.0.0"


def ensure_pillow(min_version: str = _PILLOW_MIN_VERSION) -> bool:
    """检测 Pillow 是否可用；缺失则自动 pip 安装。

    Returns:
        True 表示 Pillow 可用（已存在或安装成功）；False 表示安装失败。
    """
    # 1. 已安装则直接返回
    try:
        importlib.import_module("PIL")
        return True
    except ImportError:
        pass

    # 2. 尝试自动安装
    logger.info("[LLM Manager] 未检测到 Pillow，正在自动安装 Pillow>=%s ...", min_version)
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"Pillow>={min_version}",
                "--quiet",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
        # 安装后刷新 import 缓存并验证
        importlib.invalidate_caches()
        importlib.import_module("PIL")
        logger.info("[LLM Manager] Pillow 安装成功。")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("[LLM Manager] Pillow 安装超时（>180s），/llm list 将回退纯文本。")
    except Exception as e:  # noqa: BLE001
        logger.warning("[LLM Manager] Pillow 自动安装失败：%s，/llm list 将回退纯文本。", e)
    return False
