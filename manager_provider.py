"""LLM Manager 虚拟 Provider —— 单一注册进 AstrBot 的 Provider 实例。"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from astrbot import logger
from astrbot.api.provider import Provider
from astrbot.api.star import StarTools
from astrbot.core.provider.register import provider_cls_map

from .store import DEFAULT_MODALITIES, Store

PROVIDER_TYPE_NAME = "llm_manager"
_PLUGIN_NAME = "astrbot_plugin_llm_manager"

#: 模块级 Store 单例，由插件 Star 初始化时注入
_store: Store | None = None


def set_store(store: Store) -> None:
    """由插件主类在 __init__ 时注入共享 Store。"""
    global _store
    _store = store


def get_store() -> Store:
    """懒加载 Store（Provider 实例独立于插件 Star 被实例化时兜底）。"""
    global _store
    if _store is None:
        data_dir = StarTools.get_data_dir(_PLUGIN_NAME)
        _store = Store(data_dir / "backends.json")
    return _store


def _conversation_key(session_id, kwargs):
    conv = kwargs.get("conversation")
    umo = getattr(conv, "umo", None)
    if umo:
        return str(umo)
    if session_id:
        return str(session_id)
    return None


class LLMManagerProvider(Provider):
    def __init__(self, provider_config, provider_settings):
        Provider.__init__(self, provider_config, provider_settings)
        self._backend_cache = {}
        self.set_model(str(provider_config.get("model") or PROVIDER_TYPE_NAME))

    def get_current_key(self):
        return ""

    def set_key(self, key):
        pass

    async def get_models(self):
        seen = []
        for item in get_store().build_catalog():
            if item["model"] not in seen:
                seen.append(item["model"])
        return seen

    def _backend_for(self, source):
        key = source.get("id", "")
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached
        type_name = source.get("type", "openai_chat_completion")
        meta = provider_cls_map.get(type_name)
        if meta is None or meta.cls_type is None:
            raise RuntimeError(f"未知的提供商类型 {type_name!r}（站名：{source.get('site', '')}）。")
        cfg = get_store().source_config(source)
        cls = meta.cls_type
        try:
            backend = cls(cfg, self.provider_settings)
        except TypeError:
            backend = cls(cfg)
        self._backend_cache[key] = backend
        return backend

    def _filter_by_modalities(self, model_entry, image_urls, audio_urls, func_tool):
        mods = set(model_entry.get("modalities") or DEFAULT_MODALITIES)
        model_name = model_entry.get("name", "?")
        if image_urls and "image" not in mods:
            logger.warning("模型 %s 不支持图片输入，已自动忽略本轮图片。", model_name)
            image_urls = None
        if audio_urls and "audio" not in mods:
            logger.warning("模型 %s 不支持音频输入，已自动忽略本轮音频。", model_name)
            audio_urls = None
        if func_tool is not None and "tool_use" not in mods:
            logger.warning("模型 %s 不支持工具调用，已自动降级为纯文本对话。", model_name)
            func_tool = None
        return image_urls, audio_urls, func_tool

    def _apply_model_to_backend(self, backend, model_name):
        try:
            backend.set_model(model_name)
        except Exception:
            pass
        try:
            cfg = getattr(backend, "provider_config", None)
            if isinstance(cfg, dict):
                cfg["model"] = model_name
        except Exception:
            pass
        for attr in ("model", "_model", "model_name"):
            try:
                if hasattr(backend, attr):
                    setattr(backend, attr, model_name)
            except Exception:
                pass
        logger.info(
            "[LLM Manager] 后端模型已设置 | type=%s model_attr=%r config_model=%r",
            type(backend).__name__,
            getattr(backend, "model", "?"),
            getattr(getattr(backend, "provider_config", {}), "get", lambda k: "?")("model")
            if isinstance(getattr(backend, "provider_config", None), dict) else "?",
        )

    def _route(self, model, session_id, kwargs):
        configured_model = str(self.provider_config.get("model") or "")
        incoming_model = model
        model_ignored = False
        if model and configured_model and model == configured_model:
            model = None
            model_ignored = True
        umo = _conversation_key(session_id, kwargs)
        store = get_store()
        routed = store.resolve(req_model=model, umo=umo)
        default_inst = store.data.get("default_instance", "")
        default_model = store.data.get("default_model", "")
        override_id = store.get_conversation_override(umo) if umo else None
        logger.info(
            "[LLM Manager] 路由诊断 | incoming_model=%r placeholder=%r ignored=%s "
            "umo=%r | store default=%s/%s override=%s | routed: instance=%s model=%s",
            incoming_model, configured_model, model_ignored, umo,
            default_inst, default_model, override_id or "(无)",
            routed["instance"]["id"] if routed else "None",
            routed["model_name"] if routed else "None",
        )
        if routed is None:
            raise RuntimeError("LLM Manager 尚未配置任何可用模型。请先执行 /llm add 添加后端、/llm group add 建立分组，再用 /llm list 查看。")
        return routed

    async def text_chat(self, prompt=None, session_id=None, image_urls=None, audio_urls=None,
                        func_tool=None, contexts=None, system_prompt=None, tool_calls_result=None,
                        model=None, extra_user_content_parts=None, tool_choice="auto",
                        request_max_retries=None, **kwargs):
        routed = self._route(model, session_id, kwargs)
        backend = self._backend_for(routed["source"])
        self._apply_model_to_backend(backend, routed["model_name"])
        image_urls, audio_urls, func_tool = self._filter_by_modalities(
            routed["model_entry"], image_urls, audio_urls, func_tool)
        return await backend.text_chat(
            prompt=prompt, session_id=session_id, image_urls=image_urls, audio_urls=audio_urls,
            func_tool=func_tool, contexts=contexts, system_prompt=system_prompt,
            tool_calls_result=tool_calls_result, model=routed["model_name"],
            extra_user_content_parts=extra_user_content_parts, tool_choice=tool_choice,
            request_max_retries=request_max_retries, **kwargs,
        )

    async def text_chat_stream(self, prompt=None, session_id=None, image_urls=None, audio_urls=None,
                               func_tool=None, contexts=None, system_prompt=None, tool_calls_result=None,
                               model=None, tool_choice="auto", request_max_retries=None, **kwargs):
        routed = self._route(model, session_id, kwargs)
        backend = self._backend_for(routed["source"])
        self._apply_model_to_backend(backend, routed["model_name"])
        image_urls, audio_urls, func_tool = self._filter_by_modalities(
            routed["model_entry"], image_urls, audio_urls, func_tool)
        async for chunk in backend.text_chat_stream(
            prompt=prompt, session_id=session_id, image_urls=image_urls, audio_urls=audio_urls,
            func_tool=func_tool, contexts=contexts, system_prompt=system_prompt,
            tool_calls_result=tool_calls_result, model=routed["model_name"],
            tool_choice=tool_choice, request_max_retries=request_max_retries, **kwargs,
        ):
            yield chunk
