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

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

from astrbot.api import logger
from astrbot.api.provider import Provider
from astrbot.api.star import StarTools
from astrbot.core.provider.register import provider_cls_map

from .store import DEFAULT_MODALITIES, Store

PROVIDER_TYPE_NAME = "llm_manager"
_PLUGIN_NAME = "astrbot_plugin_llm_manager"

#: 模块级 Store 单例，由插件 Star 初始化时注入
_store: Store | None = None

#: 是否把路由诊断按 INFO 输出；False 时降为 DEBUG
_verbose_routing_log = True


def set_store(store: Store) -> None:
    """由插件主类在 __init__ 时注入共享 Store。"""
    global _store
    _store = store


def set_verbose_logging(enabled: bool) -> None:
    """由插件主类按 WebUI 配置切换路由日志级别。"""
    global _verbose_routing_log
    _verbose_routing_log = bool(enabled)


def _routing_log(message: str, *args: Any) -> None:
    """按开关输出路由诊断：开启时 INFO，关闭时 DEBUG。"""
    (logger.info if _verbose_routing_log else logger.debug)(message, *args)


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


# ---------- 故障转移：错误分类 ----------

_STATUS_IN_MESSAGE = re.compile(r"(?<![\d.])([45]\d{2})(?![\d.])")

_CONTEXT_PATTERNS = (
    "context length", "context window", "maximum context", "too many tokens",
    "reduce the length", "上下文长度", "上下文过长",
)
_CONTENT_POLICY_PATTERNS = (
    "content policy", "content_policy", "content filter", "content_filter",
    "safety", "内容政策", "违规内容", "安全策略",
)
_MODEL_MISSING_PATTERNS = (
    "model not found", "model_not_found", "unknown model", "invalid model",
    "no such model", "does not exist", "模型不存在", "模型未找到",
)
_AUTH_QUOTA_PATTERNS = (
    "invalid api key", "incorrect api key", "authentication", "unauthorized",
    "permission denied", "insufficient quota", "quota exceeded",
    "insufficient_quota", "insufficient balance", "billing",
    "欠费", "余额不足", "鉴权失败", "密钥无效",
)
_BAD_REQUEST_PATTERNS = (
    "invalid request", "invalid parameter", "bad request", "unprocessable",
    "参数错误", "请求参数",
)
_SOURCE_RETRY_PATTERNS = (
    "timeout", "timed out", "time out", "connection", "connect error",
    "network", "dns", "ssl", "econnreset", "econnrefused", "enotfound",
    "rate limit", "too many requests", "temporarily unavailable",
    "连接失败", "连接超时", "网络错误",
)
_MODEL_RETRY_PATTERNS = (
    "overloaded", "service unavailable", "internal server error",
    "bad gateway", "gateway timeout", "server error", "服务不可用",
)


def _exception_status(exc: BaseException) -> int | None:
    """尽量从异常对象里提取 HTTP 状态码。"""
    for obj in (exc, getattr(exc, "response", None)):
        if obj is None:
            continue
        if isinstance(obj, dict):
            values = [
                obj.get("status_code"), obj.get("status"), obj.get("http_status"),
            ]
        else:
            values = [
                getattr(obj, "status_code", None),
                getattr(obj, "status", None),
                getattr(obj, "http_status", None),
            ]
        for value in values:
            if value is None or isinstance(value, bool):
                continue
            try:
                code = int(value)
            except (TypeError, ValueError):
                continue
            if 100 <= code <= 599:
                return code
    return None


def _classify_error(exc: BaseException) -> tuple[bool, str, str]:
    """判断异常是否值得换模型重试。

    Returns:
        (retryable, cooldown_scope, reason)
        cooldown_scope: "source" / "model" / ""
    """
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    code = _exception_status(exc)

    # 用户取消、生成器关闭等不应重试
    if "cancel" in name or "generator exit" in name:
        return False, "", "cancelled"

    # 请求本身有问题，换模型也救不回来
    if any(p in text for p in _CONTEXT_PATTERNS):
        return False, "", "context_length"
    if any(p in text for p in _CONTENT_POLICY_PATTERNS):
        return False, "", "content_policy"

    # 模型下线/名字错误：换备选有意义
    if any(p in text for p in _MODEL_MISSING_PATTERNS):
        return True, "model", "model_not_found"

    # 鉴权/欠费：同 key 的其他模型没意义，但换 source 可能有用
    if any(p in text for p in _AUTH_QUOTA_PATTERNS):
        return True, "source", "auth_or_quota"

    if code is not None:
        if code in (400, 413, 422):
            return False, "", "bad_request"
        if code in (401, 403, 402, 408, 409, 425, 429):
            return True, "source", f"source_{code}"
        if code == 404:
            return True, "model", "model_not_found"
        if 500 <= code <= 599:
            return True, "model", f"server_{code}"

    if any(k in name for k in (
        "timeout", "connect", "network", "transport", "protocol", "dns", "ssl",
    )):
        return True, "source", "timeout_or_connection"
    if any(k in name for k in ("ratelimit", "quota")):
        return True, "source", "rate_limit"
    if "notfound" in name:
        return True, "model", "model_not_found"
    if any(k in name for k in ("authentication", "permission", "unauthorized")):
        return True, "source", "auth"
    if any(k in name for k in ("badrequest", "invalidrequest", "unprocessable")):
        return False, "", "bad_request"

    if any(p in text for p in _BAD_REQUEST_PATTERNS):
        return False, "", "bad_request"
    if any(p in text for p in _SOURCE_RETRY_PATTERNS):
        return True, "source", "timeout_or_connection"
    if any(p in text for p in _MODEL_RETRY_PATTERNS):
        return True, "model", "server_error"

    match = _STATUS_IN_MESSAGE.search(text)
    if match:
        status = int(match.group(1))
        if status in (400, 413, 422):
            return False, "", "bad_request"
        if status in (401, 403, 402, 408, 409, 425, 429):
            return True, "source", f"source_{status}"
        if status == 404:
            return True, "model", "model_not_found"
        if 500 <= status <= 599:
            return True, "model", f"server_{status}"

    return False, "", "unknown"


def _error_label(exc: BaseException) -> str:
    """日志用的错误标签，只包含异常类型和状态码，不泄露密钥。"""
    status = _exception_status(exc)
    label = type(exc).__name__
    return f"{label} status={status}" if status else label


def _route_key(routed: dict[str, Any]) -> tuple[str, str]:
    return routed["instance"]["id"], routed["model_name"]


def _describe_route(routed: dict[str, Any]) -> str:
    return f"{routed['instance']['id']}/{routed['model_name']}"


class LLMManagerProvider(Provider):
    """虚拟 Provider：注册为 `llm_manager` 类型，路由到真实后端。"""

    def __init__(self, provider_config: dict[str, Any], provider_settings: Any) -> None:
        Provider.__init__(self, provider_config, provider_settings)
        self._backend_cache: dict[tuple[str, str], Any] = {}
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
        """根据 Source 配置实例化 AstrBot 内置 Provider。

        缓存键包含完整后端配置指纹：用户修改 Key、api_base、超时、代理
        或自定义头后会自动重建，不会继续使用旧连接。
        """
        cfg = get_store().source_config(source)
        fingerprint = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
        source_id = str(cfg.get("id", ""))
        key = (source_id, fingerprint)
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached

        type_name = source.get("type", "openai_chat_completion")
        meta = provider_cls_map.get(type_name)
        if meta is None or meta.cls_type is None:
            raise RuntimeError(
                f"未知的提供商类型 {type_name!r}（站名：{source.get('site', '')}）。"
            )

        cls = meta.cls_type
        try:
            backend = cls(cfg, self.provider_settings)
        except TypeError:
            # 某些 Provider 构造函数只接受一个参数
            backend = cls(cfg)

        # 清掉同一源的旧指纹，避免 Key 轮换后缓存无限增长
        for cached_key in list(self._backend_cache):
            if (
                isinstance(cached_key, tuple)
                and cached_key[0] == source_id
                and cached_key != key
            ):
                self._backend_cache.pop(cached_key, None)
        self._backend_cache[key] = backend
        return backend

    def _record_usage(
        self,
        routed: dict[str, Any],
        usage: dict[str, int] | None,
        primary: dict[str, Any] | None = None,
    ) -> None:
        """记录一次成功调用的 token 消耗；失败尝试不记录，避免重复计数。"""
        if not usage:
            return
        fallback_from = ""
        if primary is not None and _route_key(primary) != _route_key(routed):
            fallback_from = _describe_route(primary)
        try:
            get_store().stats.record(
                model=routed["model_name"],
                instance_id=routed["instance"]["id"],
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                fallback_from=fallback_from,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[LLM Manager] 记录 token 消耗失败: %s", e)

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

        # 诊断日志：确认模型已应用（受 verbose_routing_log 开关控制）
        _routing_log(
            "[LLM Manager] 后端模型已设置 | type=%s model_attr=%r config_model=%r",
            type(backend).__name__,
            getattr(backend, "model", "?"),
            getattr(getattr(backend, "provider_config", {}), "get", lambda k: "?")("model")
            if isinstance(getattr(backend, "provider_config", None), dict) else "?",
        )

    # ---------- 路由 ----------

    def _release_stale_sticky(self, store: Any, umo: str | None) -> None:
        """撤销已失去意义的故障转移会话切换。

        sticky 只在主模型故障冷却期间有意义。一旦全局主模型不再冷却，
        继续沿用备用模型会让人误以为“降级后永远回不去”，因此在路由前
        自动清除由 fallback 写入的 override，让下一轮重新探测主模型。
        用户通过 /llmm use -c 手动设置的切换没有 origin 标记，不受影响。
        """
        if not umo:
            return
        override = store.get_conversation_override(umo)
        if not override or override.get("origin") != "fallback":
            return

        default = store.resolve(umo=None)
        if default is None:
            return
        override_key = (override.get("instance_id"), override.get("model"))
        default_key = (default["instance"]["id"], default["model_name"])
        if override_key == default_key:
            return

        source_id = (default.get("source") or {}).get("id", "")
        if store.is_cooling(
            default["instance"]["id"], default["model_name"], source_id
        ):
            return

        if store.clear_conversation_override(umo):
            logger.info(
                "[LLM Manager] 主模型冷却结束，撤销故障转移会话切换并重新探测：%s",
                _describe_route(default),
            )

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
        self._release_stale_sticky(store, umo)
        routed = store.resolve(req_model=model, umo=umo)

        # 诊断日志
        default_inst = store.data.get("default_instance", "")
        default_model = store.data.get("default_model", "")
        override = store.get_conversation_override(umo) if umo else None
        override_text = (
            f"{override['instance_id']}/{override['model']}" if override else "(无)"
        )
        _routing_log(
            "[LLM Manager] 路由诊断 | incoming_model=%r placeholder=%r ignored=%s "
            "umo=%r | store default=%s/%s override=%s | routed: instance=%s model=%s",
            incoming_model, configured_model, model_ignored, umo,
            default_inst, default_model, override_text,
            routed["instance"]["id"] if routed else "None",
            routed["model_name"] if routed else "None",
        )

        if routed is None:
            raise RuntimeError(
                "LLM Manager 尚未配置任何可用模型。"
                "请先执行 /llmm add 添加后端、/llmm group add 建立分组，再用 /llmm list 查看。"
            )
        return routed

    # ---------- 故障转移 ----------

    def _build_candidates(
        self,
        primary: dict[str, Any],
        image_urls: list[str] | None,
        audio_urls: list[str] | None,
        func_tool: Any,
    ) -> list[dict[str, Any]]:
        """主模型 + 备选池，按顺序去重；备选需满足本次请求的能力要求。"""
        store = get_store()
        fb = store.fallback_config()
        if not fb.get("enabled"):
            return [primary]

        required: set[str] = set()
        if image_urls:
            required.add("image")
        if audio_urls:
            required.add("audio")
        if func_tool is not None:
            required.add("tool_use")

        candidates = [primary]
        seen = {_route_key(primary)}
        for routed in store.resolve_fallback_entries(required):
            key = _route_key(routed)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(routed)
        return candidates

    def _prepare_plan(
        self,
        primary: dict[str, Any],
        image_urls: list[str] | None,
        audio_urls: list[str] | None,
        func_tool: Any,
    ) -> dict[str, Any]:
        """构造本次请求的候选链与冷却策略。"""
        store = get_store()
        fb = store.fallback_config()
        enabled = bool(fb.get("enabled"))
        candidates = self._build_candidates(
            primary, image_urls, audio_urls, func_tool
        )
        max_attempts = (
            max(1, int(fb.get("max_attempts") or 3)) if enabled else 1
        )
        skip_cooling = enabled

        cooling = [
            store.is_cooling(
                routed["instance"]["id"],
                routed["model_name"],
                (routed.get("source") or {}).get("id", ""),
            )
            for routed in candidates
        ]
        if enabled and candidates and all(cooling):
            if len(candidates) > 1:
                logger.warning(
                    "[LLM Manager] 所有候选均在冷却，按原顺序尝试一次：%s",
                    " -> ".join(_describe_route(c) for c in candidates),
                )
            max_attempts = 1
            skip_cooling = False

        return {
            "enabled": enabled,
            "candidates": candidates,
            "max_attempts": max_attempts,
            "skip_cooling": skip_cooling,
            "cooldown_seconds": int(fb.get("cooldown_seconds") or 0),
        }

    def _handle_failure(
        self,
        exc: BaseException,
        routed: dict[str, Any],
        cooldown_seconds: int,
        enabled: bool,
    ) -> bool:
        """记录失败并按需标记冷却，返回是否值得尝试下一个候选。"""
        retryable, scope, reason = _classify_error(exc)
        if enabled and retryable and cooldown_seconds > 0:
            source_id = (routed.get("source") or {}).get("id", "")
            get_store().mark_cooldown(
                routed["instance"]["id"],
                routed["model_name"],
                source_id,
                cooldown_seconds,
                scope,
            )
        log = logger.info if enabled else logger.debug
        log(
            "[LLM Manager] 模型调用失败 | %s | %s | 原因=%s | 可重试=%s",
            _describe_route(routed), _error_label(exc), reason, retryable,
        )
        return retryable

    def _on_success(
        self,
        primary: dict[str, Any],
        routed: dict[str, Any],
        umo: str | None,
        usage: dict[str, int] | None,
    ) -> None:
        """成功收尾：清冷却、sticky 提升、记录 token。"""
        store = get_store()
        source_id = (routed.get("source") or {}).get("id", "")
        store.clear_cooldown(
            routed["instance"]["id"], routed["model_name"], source_id
        )

        used_fallback = _route_key(primary) != _route_key(routed)
        if used_fallback:
            logger.info(
                "[LLM Manager] 故障转移成功 | %s -> %s",
                _describe_route(primary), _describe_route(routed),
            )
            if store.fallback_config().get("mode") == "sticky" and umo:
                current = store.get_conversation_override(umo)
                # 用户手动 /llmm use -c 选择的会话不自动粘滞，避免覆盖用户意图。
                if current is None or current.get("origin") == "fallback":
                    target = (
                        f"{routed['instance']['id']}/{routed['model_name']}"
                    )
                    try:
                        if store.set_conversation_override(
                            umo, target, origin="fallback"
                        ):
                            logger.info(
                                "[LLM Manager] sticky 已把当前会话切换到 %s", target
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[LLM Manager] sticky 会话切换失败：%s", e)

        self._record_usage(routed, usage, primary)

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
        primary = self._route(model, session_id, kwargs)
        store = get_store()
        plan = self._prepare_plan(primary, image_urls, audio_urls, func_tool)
        umo = _conversation_key(session_id, kwargs)

        base_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "session_id": session_id,
            "contexts": contexts,
            "system_prompt": system_prompt,
            "tool_calls_result": tool_calls_result,
            "extra_user_content_parts": extra_user_content_parts,
            "tool_choice": tool_choice,
            "request_max_retries": request_max_retries,
            **kwargs,
        }

        attempted = 0
        last_exc: BaseException | None = None

        for index, routed in enumerate(plan["candidates"]):
            if attempted >= plan["max_attempts"]:
                break
            source_id = (routed.get("source") or {}).get("id", "")
            # 候选链首位始终探测一次：主模型恢复后能立刻回到主模型，
            # 不会因为残留冷却被长期跳过。冷却只用于跳过后续备选。
            if (
                index > 0
                and plan["skip_cooling"]
                and store.is_cooling(
                    routed["instance"]["id"], routed["model_name"], source_id
                )
            ):
                logger.debug(
                    "[LLM Manager] 跳过冷却中的候选：%s", _describe_route(routed)
                )
                continue

            attempted += 1
            try:
                backend = self._backend_for(routed["source"])
                self._apply_model_to_backend(backend, routed["model_name"])
                cur_images, cur_audio, cur_tools = self._filter_by_modalities(
                    routed["model_entry"], image_urls, audio_urls, func_tool
                )
                response = await backend.text_chat(
                    image_urls=cur_images,
                    audio_urls=cur_audio,
                    func_tool=cur_tools,
                    model=routed["model_name"],
                    **base_kwargs,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                retryable = self._handle_failure(
                    e, routed, plan["cooldown_seconds"], plan["enabled"]
                )
                if not retryable or attempted >= plan["max_attempts"]:
                    raise
                last_exc = e
                continue

            try:
                usage = _extract_usage(response)
            except Exception:  # noqa: BLE001
                usage = None
            self._on_success(primary, routed, umo, usage)
            return response

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM Manager 的候选模型均处于冷却，请稍后重试。")

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
        primary = self._route(model, session_id, kwargs)
        store = get_store()
        plan = self._prepare_plan(primary, image_urls, audio_urls, func_tool)
        umo = _conversation_key(session_id, kwargs)

        base_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "session_id": session_id,
            "contexts": contexts,
            "system_prompt": system_prompt,
            "tool_calls_result": tool_calls_result,
            "tool_choice": tool_choice,
            "request_max_retries": request_max_retries,
            **kwargs,
        }

        attempted = 0
        last_exc: BaseException | None = None

        for index, routed in enumerate(plan["candidates"]):
            if attempted >= plan["max_attempts"]:
                break
            source_id = (routed.get("source") or {}).get("id", "")
            # 首位候选始终探测一次，避免主模型恢复后仍被冷却长期跳过。
            if (
                index > 0
                and plan["skip_cooling"]
                and store.is_cooling(
                    routed["instance"]["id"], routed["model_name"], source_id
                )
            ):
                logger.debug(
                    "[LLM Manager] 跳过冷却中的候选：%s", _describe_route(routed)
                )
                continue

            attempted += 1
            usage: dict[str, int] | None = None
            visible = False
            try:
                backend = self._backend_for(routed["source"])
                self._apply_model_to_backend(backend, routed["model_name"])
                cur_images, cur_audio, cur_tools = self._filter_by_modalities(
                    routed["model_entry"], image_urls, audio_urls, func_tool
                )
                async for chunk in backend.text_chat_stream(
                    image_urls=cur_images,
                    audio_urls=cur_audio,
                    func_tool=cur_tools,
                    model=routed["model_name"],
                    **base_kwargs,
                ):
                    # 空 chunk 不转发，也不算“已经输出”，仍允许故障转移
                    if isinstance(chunk, str):
                        if not chunk:
                            continue
                        visible = True
                    else:
                        # 非字符串 chunk（工具调用/用量）一旦出现就不再切换
                        visible = True
                        try:
                            candidate = _extract_usage(chunk)
                        except Exception:  # noqa: BLE001
                            candidate = None
                        if candidate:
                            usage = candidate
                    yield chunk
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                retryable = self._handle_failure(
                    e, routed, plan["cooldown_seconds"], plan["enabled"]
                )
                # 已经向用户输出过内容后不再切换，避免两段拼接；其余情况交给调用方
                if visible or not retryable or attempted >= plan["max_attempts"]:
                    raise
                last_exc = e
                continue
            else:
                self._on_success(primary, routed, umo, usage)
                return

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM Manager 的候选模型均处于冷却，请稍后重试。")
