"""AstrBot Provider 适配器注册工具（幂等）。

参考社区插件 astrbot_plugin_openai_responses 的注册方式：
- 直接操作 astrbot.core.provider.register 的 provider_cls_map / provider_registry；
- 支持插件热重载场景下的替换注册，避免重名冲突。
"""
from __future__ import annotations

from typing import Any

from astrbot import logger
from astrbot.core.provider.entities import ProviderMetaData, ProviderType
from astrbot.core.provider.register import provider_cls_map, provider_registry


def register_or_replace_provider_adapter(
    *,
    provider_type_name: str,
    desc: str,
    cls_type: type[Any],
    provider_type: ProviderType = ProviderType.CHAT_COMPLETION,
    default_config_tmpl: dict | None = None,
    provider_display_name: str | None = None,
) -> ProviderMetaData:
    """注册 Provider 适配器；若同名已注册则替换（支持插件热重载）。"""
    if default_config_tmpl is not None:
        default_config_tmpl = dict(default_config_tmpl)
        default_config_tmpl.setdefault("type", provider_type_name)
        default_config_tmpl.setdefault("enable", False)
        default_config_tmpl.setdefault("id", provider_type_name)

    removed = 0
    for item in list(provider_registry):
        if getattr(item, "type", None) == provider_type_name:
            provider_registry.remove(item)
            removed += 1

    pm = ProviderMetaData(
        id="default",  # 实例化时会被替换为真实实例 id
        model=None,
        type=provider_type_name,
        desc=desc,
        provider_type=provider_type,
        cls_type=cls_type,
        default_config_tmpl=default_config_tmpl,
        provider_display_name=provider_display_name,
    )
    provider_cls_map[provider_type_name] = pm
    provider_registry.append(pm)

    if removed:
        logger.info(
            "Provider adapter registration replaced: %s (removed %s old entries)",
            provider_type_name,
            removed,
        )
    else:
        logger.info("Provider adapter registered: %s", provider_type_name)
    return pm


def unregister_provider_adapter(provider_type_name: str) -> None:
    """注销 Provider 适配器（插件卸载/停用时调用）。"""
    provider_cls_map.pop(provider_type_name, None)
    for item in list(provider_registry):
        if getattr(item, "type", None) == provider_type_name:
            provider_registry.remove(item)
    logger.info("Provider adapter unregistered: %s", provider_type_name)
