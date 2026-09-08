"""LLM Manager 虚拟 Provider —— 单一注册进 AstrBot 的 Provider 实例。

请求进来后按路由规则选择真实后端（Source），再用 AstrBot 内置 Provider 类
（从 provider_cls_map 按 type 解析）实例化并委托调用，完整继承各家 API 协议、
工具调用与流式能力。新增后端/模型分组只需改插件自己的 backends.json。
"""
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


def _conversation_key(session_id: str | None, kwargs: dict[str, Any]) -> str | None:
    """尽力从请求中提取会话键（umo）。ProviderRequest.conversation 可能携带 umo。"""
    conv = kwargs.get("conversation")
    umo = getattr(conv, "umo", None)
    if umo:
        return str(umo)
    if session_id:
        return str(session_id)
    return None


class LLMManagerProvider(Provider):
    """虚拟聚合 Provider：路由 + 委托。"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        Provider.__init__(self, provider_config, provider_settings)
        # 注意：不缓存 get_store()！AstrBot 启动时 Provider 实例化可能先于
        # 插件 Star 类 __init__，缓存会导致指令层切换的 store 与 Provider 路由
        # 用的 store 不是同一个对象。每次路由时实时 get_store() 取模块级单例。
        self._backend_cache: dict[str, Provider] = {}
        self.set_model(str(provider_config.get("model") or PROVIDER_TYPE_NAME))

    # ---------- 基类抽象方法 ----------

    def get_current_key(self) -> str:
        return ""

    def set_key(self, key: str) -> None:
        pass

    async def get_models(self) -> list[str]:
        seen: list[str] = []
        for item in get_store().build_catalog():
            if item["model"] not in seen:
                seen.append(item["model"])
        return seen

    # ---------- 后端实例化与缓存 ----------

    def _backend_for(self, source: dict[str, Any]) -> Provider:
        key = source.get("id", "")
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached

        type_name = source.get("type", "openai_chat_completion")
        meta = provider_cls_map.get(type_name)
        if meta is None or meta.cls_type is None:
            raise RuntimeError(
                f"未知的提供商类型 {type_name!r}（站名：{source.get('site', '')}）。"
                "请检查 /llm add 时填写的类型是否存在于 AstrBot。"
            )

        cfg = get_store().source_config(source)
        cls = meta.cls_type
        try:
            backend = cls(cfg, self.provider_settings)
        except TypeError:
            backend = cls(cfg)  # 兼容部分单参数实现的适配器
        self._backend_cache[key] = backend
        return backend

    # ---------- 能力适配 ----------

    def _filter_by_modalities(
        self,
        model_entry: dict[str, Any],
        image_urls: list[str] | None,
        audio_urls: list[str] | None,
        func_tool: Any,
    ) -> tuple[list[str] | None, list[str] | None, Any]:
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

    # ---------- 后端模型强制设置 ----------

    def _apply_model_to_backend(self, backend: Provider, model_name: str) -> None:
        """AstrBot 内置 Provider 可能从多处读取模型，逐一覆盖确保生效。"""
        # 1. 标准 set_model
        try:
            backend.set_model(model_name)
        except Exception:
            pass
        # 2. provider_config["model"]（部分 Provider 从配置读）
        try:
            cfg = getattr(backend, "provider_config", None)
            if isinstance(cfg, dict):
                cfg["model"] = model_name
        except Exception:
            pass
        # 3. 直接属性覆盖
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

    # ---------- 对话入口 ----------

    def _route(
        self,
        model: str | None,
        session_id: str | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        # AstrBot 调用时会把本 Provider 实例配置里的 model 字段传进来。
        # 该字段是占位符（如 "main"），不是真实模型名；若与配置值一致则忽略，
        # 让路由走全局默认 / 会话 override。只有显式指定的真实模型名才被尊重。
        configured_model = str(self.provider_config.get("model") or "")
        incoming_model = model
        model_ignored = False
        if model and configured_model and model == configured_model:
            model = None
            model_ignored = True

        umo = _conversation_key(session_id, kwargs)
        store = get_store()
        routed = store.resolve(req_model=model, umo=umo)

        # 详细路由日志，用于排查"切换不生效"
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
            raise RuntimeError(
                "LLM Manager 尚未配置任何可用模型。请先执行 /llm add 添加后端、"
                "/llm group add 建立分组，再用 /llm list 查看。"
            )
        return routed

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: Any = None,
        contexts: list[dict] | list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any = None,
        model: str | None = None,
        extra_user_content_parts: list[Any] | None = None,
        tool_choice: str = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        routed = self._route(model, session_id, kwargs)
        backend = self._backend_for(routed["source"])
        self._apply_model_to_backend(backend, routed["model_name"])
        image_urls, audio_urls, func_tool = self._filter_by_modalities(
            routed["model_entry"], image_urls, audio_urls, func_tool
        )
        return await backend.text_chat(
            prompt=prompt,
            session_id=session_id,
            image_urls=image_urls,
            audio_urls=audio_urls,
            func_tool=func_tool,
            contexts=contexts,
            system_prompt=system_prompt,
            tool_calls_result=tool_calls_result,
            model=routed["model_name"],
            extra_user_content_parts=extra_user_content_parts,
            tool_choice=tool_choice,
            request_max_retries=request_max_retries,
            **kwargs,
        )

    async def text_chat_stream(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: Any = None,
        contexts: list[dict] | list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any = None,
        model: str | None = None,
        tool_choice: str = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        routed = self._route(model, session_id, kwargs)
        backend = self._backend_for(routed["source"])
        self._apply_model_to_backend(backend, routed["model_name"])
        image_urls, audio_urls, func_tool = self._filter_by_modalities(
            routed["model_entry"], image_urls, audio_urls, func_tool
        )
        async for chunk in backend.text_chat_stream(
            prompt=prompt,
            session_id=session_id,
            image_urls=image_urls,
            audio_urls=audio_urls,
            func_tool=func_tool,
            contexts=contexts,
            system_prompt=system_prompt,
            tool_calls_result=tool_calls_result,
            model=routed["model_name"],
            tool_choice=tool_choice,
            request_max_retries=request_max_retries,
            **kwargs,
        ):
            yield chunk
