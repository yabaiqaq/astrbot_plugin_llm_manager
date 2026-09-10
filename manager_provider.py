"""LLM Manager 虚拟 Provider —— 单一注册进 AstrBot 的 Provider 实例。

职责：
  - 注册为 AstrBot 的 provider 类型 `llm_manager`，由用户在 WebUI 建一个实例。
  - 收到请求时，根据当前全局默认 / 会话切换 / 请求显式模型，从 Store 路由到真实后端。
  - 动态实例化 AstrBot 内置 Provider（如 openai_chat_completion）并委托调用。
  - 按模型能力自动降级（图片/音频/工具调用不支持时忽略或降级）。

设计要点：
  - Store 单例由插件 Star 初始化时通过 set_store() 注入；
    Provider 独立被 AstrBot 实例化时走懒加载兜底。
  - 后端实例按 source id 缓存，避免每次请求都新建。
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
    """从会话上下文提取统一消息源（umo），用于会话级切换。"""
    conv = kwargs.get("conversation")
    umo = getattr(conv, "umo", None)
    if umo:
        return str(umo)
    if session_id:
        return str(session_id)
    return None


def _parse_usage_obj(usage: Any) -> dict[str, int] | None:
    """从 usage 对象中解析 prompt_tokens/completion_tokens/total_tokens。

    兼容多种字段命名（OpenAI 标准 / Gemini / Anthropic 等）。
    """
    prompt = None
    completion = None
    total = None

    for attr in ("prompt_tokens", "input_tokens", "input", "prompt_token_count"):
        val = getattr(usage, attr, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(attr)
        if val is not None:
            try:
                prompt = int(val)
                break
            except (TypeError, ValueError):
                pass

    for attr in ("completion_tokens", "output_tokens", "output", "candidates_token_count"):
        val = getattr(usage, attr, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(attr)
        if val is not None:
            try:
                completion = int(val)
                break
            except (TypeError, ValueError):
                pass

    for attr in ("total_tokens", "total", "total_token_count"):
        val = getattr(usage, attr, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(attr)
        if val is not None:
            try:
                total = int(val)
                break
            except (TypeError, ValueError):
                pass

    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)

    if prompt is None and completion is None and total is None:
        return None

    return {
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "total_tokens": total or 0,
    }


def _extract_usage(response: Any) -> dict[str, int] | None:
    """从 Provider 响应中提取 token 用量。

    尝试多种可能的属性路径：
      - response.usage / token_usage / usage_metadata
      - response.raw_completion.usage
      - response 本身就是 usage 对象
    """
    if isinstance(response, str):
        return None

    # 1. 直接从响应对象的 usage 类属性提取
    for attr in ("usage", "token_usage", "usage_metadata"):
        usage = getattr(response, attr, None)
        if usage is not None:
            result = _parse_usage_obj(usage)
            if result:
                return result

    # 2. 从 raw_completion 提取
    raw = getattr(response, "raw_completion", None)
    if raw is not None:
        for attr in ("usage", "usage_metadata"):
            usage = getattr(raw, attr, None)
            if usage is not None:
                result = _parse_usage_obj(usage)
                if result:
                    return result

    # 3. 响应对象本身可能就是 usage（兜底）
    result = _parse_usage_obj(response)
    if result:
        return result

    return None


class LLMManagerProvider(Provider):
    """虚拟 Provider：注册为 `llm_manager` 类型，路由到真实后端。"""

    def __init__(self, provider_config: dict[str, Any], provider_settings: Any) -> None:
        Provider.__init__(self, provider_config, provider_settings)
        self._backend_cache: dict[str, Any] = {}
        # model 字段是占位符（用户在 WebUI 填什么都行，实际模型由 Store 路由决定）
        self.set_model(str(provider_config.get("model") or PROVIDER_TYPE_NAME))

    # ---------- Provider 接口（AstrBot 要求实现）----------

    def get_current_key(self) -> str:
        return ""

    def set_key(self, key: str) -> None:
        # 虚拟 Provider 不直接持有 key，真实 key 在各 Source 中
        pass

    async def get_models(self) -> list[str]:
        """返回当前可用的模型名列表（去重，按目录顺序）。"""
        seen: list[str] = []
        for item in get_store().build_catalog():
            if item["model"] not in seen:
                seen.append(item["model"])
        return seen

    # ---------- 后端实例化与缓存 ----------

    def _backend_for(self, source: dict[str, Any]) -> Any:
        """根据 Source 配置实例化 AstrBot 内置 Provider，按 source id 缓存。"""
        key = source.get("id", "")
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached

        type_name = source.get("type", "openai_chat_completion")
        meta = provider_cls_map.get(type_name)
        if meta is None or meta.cls_type is None:
            raise RuntimeError(
                f"未知的提供商类型 {type_name!r}（站名：{source.get('site', '')}）。"
            )

        cfg = get_store().source_config(source)
        cls = meta.cls_type
        try:
            backend = cls(cfg, self.provider_settings)
        except TypeError:
            # 某些 Provider 构造函数只接受一个参数
            backend = cls(cfg)

        self._backend_cache[key] = backend
        return backend

    # ---------- 能力降级 ----------

    def _filter_by_modalities(
        self,
        model_entry: dict[str, Any],
        image_urls: list[str] | None,
        audio_urls: list[str] | None,
        func_tool: Any,
    ) -> tuple[list[str] | None, list[str] | None, Any]:
        """根据模型能力声明，自动降级不支持的输入类型。"""
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

    # ---------- 模型应用（强制覆盖后端 model 字段）----------

    def _apply_model_to_backend(self, backend: Any, model_name: str) -> None:
        """把路由到的模型名强制应用到后端实例，防止后端用配置里的旧模型。

        多管齐下：set_model() + provider_config["model"] + 实例属性。
        """
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

        # 诊断日志：确认模型已应用
        logger.info(
            "[LLM Manager] 后端模型已设置 | type=%s model_attr=%r config_model=%r",
            type(backend).__name__,
            getattr(backend, "model", "?"),
            getattr(getattr(backend, "provider_config", {}), "get", lambda k: "?")("model")
            if isinstance(getattr(backend, "provider_config", None), dict) else "?",
        )

    # ---------- 路由 ----------

    def _route(
        self,
        model: str | None,
        session_id: str | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """核心路由：根据 请求模型 / 会话切换 / 全局默认 解析到真实后端。

        返回 {source, instance, model_name, model_entry}。
        未配置任何模型时抛 RuntimeError（AstrBot 会捕获并提示用户）。
        """
        # 判断请求里的 model 是否是 LLM Manager 实例配置里的占位符
        # 如果是，说明 AstrBot 把配置里的 model 传进来了，应该忽略它（用 Store 路由）
        configured_model = str(self.provider_config.get("model") or "")
        incoming_model = model
        model_ignored = False
        if model and configured_model and model == configured_model:
            model = None
            model_ignored = True

        umo = _conversation_key(session_id, kwargs)
        store = get_store()
        routed = store.resolve(req_model=model, umo=umo)

        # 诊断日志
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
                "LLM Manager 尚未配置任何可用模型。"
                "请先执行 /llm add 添加后端、/llm group add 建立分组，再用 /llm list 查看。"
            )
        return routed

    # ---------- 文本对话（非流式）----------

    async def text_chat(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: Any = None,
        contexts: list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any = None,
        model: str | None = None,
        extra_user_content_parts: Any = None,
        tool_choice: str = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> str:
        routed = self._route(model, session_id, kwargs)
        backend = self._backend_for(routed["source"])
        self._apply_model_to_backend(backend, routed["model_name"])

        image_urls, audio_urls, func_tool = self._filter_by_modalities(
            routed["model_entry"], image_urls, audio_urls, func_tool
        )

        response = await backend.text_chat(
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

        # 记录 token 消耗（不影响主流程，失败仅记日志）
        try:
            usage = _extract_usage(response)
            if usage:
                get_store().stats.record(
                    model=routed["model_name"],
                    instance_id=routed["instance"]["id"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("[LLM Manager] 记录 token 消耗失败: %s", e)

        return response

    # ---------- 文本对话（流式）----------

    async def text_chat_stream(
        self,
        prompt: str | None = None,
        session_id: str | None = None,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        func_tool: Any = None,
        contexts: list[Any] | None = None,
        system_prompt: str | None = None,
        tool_calls_result: Any = None,
        model: str | None = None,
        tool_choice: str = "auto",
        request_max_retries: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
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
