"""LLM 供应商统一管理插件主入口。

设计：AstrBot 中只注册一个虚拟 Provider（LLM Manager），全部真实后端、
分组与模型由插件自己的 backends.json 管理；指令层提供三级查看与全局切换。

指令集（全部以 /llmm 开头）：
  list    三级查看（默认图片卡片，-t 纯文本）
  search  按模型名关键词模糊搜索
  use     全局/会话切换（序号/模型ID/实例ID）
  default 查看/设置全局默认
  add     新增一级源（站名 + base_url + key）
  import  从系统配置(cmd_config.json)导入已有供应商
  resync  从系统配置完全重建（解决手动删除后不同步）
  group   分组实例管理（add/rm/enable/disable）
  model   模型挂载管理（add/rm）
  rm      统一删除（source/instance/model）
  fallback 备选池与故障转移（add/rm/move/clear/on/off/mode/max/cooldown/check）
  enable/disable  启停实例
  test    连通性测试
  help    帮助
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from .manager_provider import (
    LLMManagerProvider,
    PROVIDER_TYPE_NAME,
    get_store,
    set_store,
    set_verbose_logging,
)
from .register_provider import (
    register_or_replace_provider_adapter,
    unregister_provider_adapter,
)
from .store import DEFAULT_MODALITIES, Store

# Pillow 仅影响卡片图片，不应阻断插件加载；缺失时回退纯文本。
try:
    from .renderer import render_catalog_image
except Exception as e:  # noqa: BLE001
    render_catalog_image = None
    logger.warning("Pillow/renderer 不可用，卡片图片将回退纯文本：%s", e)

PLUGIN_NAME = "astrbot_plugin_llm_manager"
PROVIDER_TEMPLATE_KEY = "LLM Manager"


def _provider_template() -> dict:
    """WebUI 中 LLM Manager 提供商实例的默认配置模板。"""
    return {
        "id": PROVIDER_TYPE_NAME,
        "type": PROVIDER_TYPE_NAME,
        "provider": PROVIDER_TYPE_NAME,
        "provider_type": "chat_completion",
        "enable": True,
        "key": [],
        "api_base": "",
        "timeout": 120,
        "proxy": "",
        "custom_headers": {},
        # 虚拟 Provider 作为路由器对外声明完整能力，
        # 真实模型不支持时由 _filter_by_modalities 降级。
        "modalities": ["text", "tool_use", "image", "audio"],
    }


def _inject_provider_source_template(template_key: str, template: dict) -> None:
    """把 LLM Manager 模板注入 AstrBot WebUI 的 provider_sources 配置模板，
    使用户能在 WebUI 看到并创建 LLM Manager 实例。"""
    try:
        from astrbot.core.config.default import CONFIG_METADATA_2

        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template[template_key] = copy.deepcopy(template)
    except Exception as e:
        logger.warning("注入 WebUI 提供商模板失败（不影响功能）: %s", e)


def _remove_provider_source_template(template_key: str) -> None:
    try:
        from astrbot.core.config.default import CONFIG_METADATA_2

        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template.pop(template_key, None)
    except Exception:
        return


def _register_provider() -> None:
    """把 LLMManagerProvider 注册为 AstrBot 的 provider 类型。"""
    template = _provider_template()
    register_or_replace_provider_adapter(
        provider_type_name=PROVIDER_TYPE_NAME,
        desc="LLM 供应商统一管理（路由/委托）",
        cls_type=LLMManagerProvider,
        default_config_tmpl=template,
        provider_display_name=PROVIDER_TEMPLATE_KEY,
    )
    _inject_provider_source_template(PROVIDER_TEMPLATE_KEY, template)


# 模块加载时立即注册（AstrBot 加载插件时执行）
_register_provider()


def _command_argv(event: AstrMessageEvent, subcommand: str, fallback: str = "") -> list[str]:
    """解析 `/llmm <subcommand>` 之后的完整参数列表。

    AstrBot 会把消息按空格逐个映射到位置参数，单个 `arg: str` 只能拿到
    第一个词。这里优先从原始消息文本解析，保证 add/group/rm 等多参数
    指令完整可用；直接调用（单元测试或旧版本）时退回位置参数。
    """
    tokens = str(getattr(event, "message_str", "") or "").strip().split()
    sub = subcommand.casefold()
    if tokens and tokens[0].lstrip("/／#!").casefold() == sub:
        # 某些调用方会先剥掉 `/llmm`，只留下子命令及参数
        return tokens[1:]
    for i in range(len(tokens) - 1):
        head = tokens[i].lstrip("/／#!").casefold()
        if head.endswith("llmm") and tokens[i + 1].casefold() == sub:
            return tokens[i + 2:]
    return fallback.split()


def _modality_mark(modalities: list[str] | None) -> str:
    """模型能力标记（工具/视觉/语音），用于纯文本列表显示。"""
    marks = []
    if "tool_use" in (modalities or DEFAULT_MODALITIES):
        marks.append("工具")
    if "image" in (modalities or []):
        marks.append("视觉")
    if "audio" in (modalities or []):
        marks.append("语音")
    return ("·" + "/".join(marks)) if marks else ""


def _render_tree(store: Store, umo: str | None = None) -> str:
    """纯文本三级树渲染（/llmm list -t 使用）。"""
    lines: list[str] = []

    override = store.get_conversation_override(umo) if umo else None
    override_id = override["instance_id"] if override else ""
    default_inst = store.data.get("default_instance", "")
    default_model = store.data.get("default_model", "")

    # 当前生效的模型（用于标记 ● 使用中）
    current_routed = store.resolve(umo=umo)
    cur_instance_id = current_routed["instance"]["id"] if current_routed else ""
    cur_model_name = current_routed["model_name"] if current_routed else ""

    # 按 api_base 分组显示（与图片渲染一致）
    api_groups: dict[str, list[dict]] = {}
    for src in store.sources():
        api_base = str(src.get("api_base", ""))
        api_groups.setdefault(api_base, []).append(src)
    api_groups = dict(sorted(api_groups.items(), key=lambda x: x[0]))

    for api_base, sources in api_groups.items():
        lines.append(f"🔗 {api_base}  ({len(sources)} 个源)")
        for src in sources:
            src_status = "（停用）" if not src.get("enabled", True) else ""
            lines.append(f" └─ {src.get('site', '?')}  [{src.get('type', '?')}]{src_status}")
            for inst in src.get("instances", []):
                inst_id = inst.get("id", "")
                marks = []
                if inst.get("enabled", True) is False:
                    marks.append("停用")
                elif not src.get("enabled", True):
                    marks.append("源停用")
                if inst_id == default_inst and default_model == "":
                    marks.append("← 默认")
                if override and inst_id == override_id:
                    marks.append("← 本会话")
                mark_txt = f"  [{' '.join(marks)}]" if marks else ""
                lines.append(f"    └─ {inst_id}{mark_txt}")

                models = inst.get("models", [])
                if not models:
                    lines.append("        （暂无模型，用 /llmm model add 挂载）")
                    continue
                for item in store.build_catalog():
                    if item["instance_id"] != inst_id:
                        continue
                    dis = "（停用）" if item["disabled"] else ""
                    cur = ""
                    if not dis and inst_id == cur_instance_id and item["model"] == cur_model_name:
                        cur = "  ● 使用中"
                    lines.append(
                        f"        ├─ [{item['num']}] {item['model']}"
                        f"{_modality_mark(item['modalities'])}{dis}{cur}"
                    )

    if not store.sources():
        lines.append(
            "（尚未配置任何后端。示例：/llmm add DeepSeek https://api.deepseek.com/v1 sk-xxx）"
        )

    total = len(store.build_catalog())
    lines.append(f"—— 共 {len(store.sources())} 个源 / {total} 个模型 ——")
    return "\n".join(lines)


def _fallback_usage() -> str:
    return (
        "用法：\n"
        "  /llmm fallback                       查看开关、模式与备选池\n"
        "  /llmm fallback add <目标...>          追加备选（目录序号/实例/模型）\n"
        "  /llmm fallback rm <池序号...>         删除，支持 2-4 区间\n"
        "  /llmm fallback move <池序号> <新位置> 调整优先级\n"
        "  /llmm fallback clear confirm          清空并关闭\n"
        "  /llmm fallback on|off                 启用/停用\n"
        "  /llmm fallback mode transient|sticky  切换模式\n"
        "  /llmm fallback max 2-5                总尝试次数（含主模型）\n"
        "  /llmm fallback cooldown 0-3600        冷却秒数，0 关闭\n"
        "  /llmm fallback check                  静态校验，不调用模型"
    )


def _parse_pool_indexes(tokens: list[str]) -> tuple[list[int], list[str]]:
    """解析池序号，支持 `1`、`1 3`、`2-4`、`1,3`；返回 (序号, 无效项)。"""
    indexes: list[int] = []
    invalid: list[str] = []
    for token in tokens:
        for part in str(token).replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                left, _, right = part.partition("-")
                left, right = left.strip(), right.strip()
                if not (left.isdigit() and right.isdigit()):
                    invalid.append(part)
                    continue
                start, end = int(left), int(right)
                if start <= 0 or end < start or end - start > 100:
                    invalid.append(part)
                    continue
                for number in range(start, end + 1):
                    if number not in indexes:
                        indexes.append(number)
            elif part.isdigit() and int(part) > 0:
                number = int(part)
                if number not in indexes:
                    indexes.append(number)
            else:
                invalid.append(part)
    return indexes, invalid


def _render_fallback(store: Store, umo: str | None = None) -> str:
    """渲染故障转移配置与备选池状态。"""
    fb = store.fallback_config()
    enabled = bool(fb.get("enabled"))
    mode = str(fb.get("mode") or "transient")
    attempts = int(fb.get("max_attempts") or 3)
    cooldown = int(fb.get("cooldown_seconds") or 0)

    default = store.resolve(umo=None)
    primary_text = (
        f"{default['instance']['id']}/{default['model_name']}"
        if default else "（尚未配置）"
    )
    mode_text = (
        "仅当前会话粘滞，主模型冷却结束后自动恢复"
        if mode == "sticky"
        else "每轮先试主模型，失败才临时降级"
    )
    cooldown_text = f"{cooldown} 秒" + ("（已关闭）" if cooldown == 0 else "")

    lines = [
        f"故障转移：{'开启' if enabled else '关闭'}"
        + ("" if enabled else "（备选池保留，/llmm fallback on 开启）"),
        f"模式：{mode}（{mode_text}）",
        f"主模型：{primary_text}",
        f"尝试上限：{attempts}（含主模型） · 冷却：{cooldown_text}",
    ]

    if default:
        remaining = store.cooldown_remaining(
            default["instance"]["id"],
            default["model_name"],
            (default.get("source") or {}).get("id", ""),
        )
        if remaining > 0:
            lines.append(f"主模型冷却：剩余 {remaining} 秒")

    override = store.get_conversation_override(umo) if umo else None
    if override and override.get("model"):
        origin = override.get("origin")
        label = "故障转移粘滞" if origin == "fallback" else "手动切换"
        lines.append(
            f"当前会话：{override['instance_id']}/{override['model']}（{label}）"
        )
        lines.append("  恢复全局默认：/llmm use global")
        if origin == "fallback":
            lines.append("  主模型冷却结束后会自动恢复。")

    statuses = store.fallback_status()
    if not statuses:
        lines.extend([
            "",
            "备选池为空。用 /llmm fallback add <目录序号> 添加。",
        ])
        return "\n".join(lines)

    lines.extend(["", f"备选池（{len(statuses)}）："])
    has_bad = False
    for item in statuses:
        num = f"  [{item['num']}]" if item["num"] is not None else ""
        if item["status"] == "missing":
            status = "失效（实例或模型已删除）"
            has_bad = True
        elif item["status"] == "disabled":
            status = "停用（运行时跳过）"
            has_bad = True
        elif item["cooldown"] > 0:
            status = f"冷却中 {item['cooldown']}s"
        else:
            status = "可用"
        lines.append(
            f"  {item['index']}. {item['instance_id']}/{item['model']}"
            f"{num}  {status}"
        )
    if has_bad:
        lines.append("失效/停用条目会被跳过，可用 /llmm fallback check 查看详情。")
    return "\n".join(lines)


def _resolve_and_apply(
    store: Store,
    event: AstrMessageEvent,
    target: str,
    conv_only: bool,
) -> str:
    """解析切换目标并应用（全局或会话级）。"""
    if target == "global":
        umo = event.unified_msg_origin
        cleared = store.clear_conversation_override(umo)
        cur = store.data.get("default_instance", "")
        return (
            f"已清除本会话切换，回到全局默认：{cur or '（未设置）'}"
            if cleared
            else f"本会话本就没有独立切换，全局默认：{cur or '（未设置）'}"
        )

    item = store.find_in_catalog(target)
    if item is None:
        return (
            f"未找到目标 {target!r}。可用：模型序号（/llmm list 查看）、模型 ID、"
            "实例 ID（站名_分组名）。"
        )

    umo = event.unified_msg_origin
    if conv_only:
        store.set_conversation_override(umo, target)
        return f"本会话已切换到 [{item['num']}] {item['model']}（{item['instance_id']}）。"

    store.set_default(target)
    return f"已全局切换到 [{item['num']}] {item['model']}（{item['instance_id']}）。"


@register(PLUGIN_NAME, "yabaiqaq", "LLM 供应商三级管理插件：统一后端、分组与模型切换", "3.0.2")
class LLMManagerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        store = Store(data_dir / "backends.json")
        set_store(store)
        set_verbose_logging(bool(config.get("verbose_routing_log", True)))
        logger.info("LLM Manager 插件已加载，配置仓库：%s", data_dir / "backends.json")

    async def terminate(self) -> None:
        unregister_provider_adapter(PROVIDER_TYPE_NAME)
        _remove_provider_source_template(PROVIDER_TEMPLATE_KEY)

    # ---------- 权限 ----------

    def _is_bot_admin(self, event: AstrMessageEvent) -> bool:
        """判断发送者是否为 AstrBot 管理员。"""
        # 优先用 event.is_admin()（AstrBot 内置判断）
        try:
            attr = getattr(event, "is_admin", None)
            if callable(attr):
                if attr():
                    return True
            elif attr:
                return True
        except Exception:
            pass
        # 兜底：从 bot_config.admins_id 判断
        try:
            sender_id = str(event.get_sender_id())
            bot_config = None
            fn = getattr(self.context, "get_bot_config", None)
            if callable(fn):
                bot_config = fn()
            if bot_config is None:
                bot_config = getattr(self.context, "astrbot_config", None) or {}
            admins_id = bot_config.get("admins_id", []) or []
            return sender_id in [str(a) for a in admins_id]
        except Exception:
            return False

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        """根据插件配置 admin_only 判断是否允许执行。"""
        if not self.config.get("admin_only", True):
            return True
        return self._is_bot_admin(event)

    def _deny_if_not_admin(self, event: AstrMessageEvent):
        """非管理员时返回拒绝消息，管理员返回 None。"""
        if not self._is_allowed(event):
            return event.plain_result(
                "/llmm 指令仅管理员可用。如需放开，请在插件配置中关闭 admin_only。"
            )
        return None

    # ---------- 指令组 ----------

    @filter.command_group("llmm")
    def llmm(self):
        pass

    # ---------- /llmm list ----------

    @llmm.command("list")
    async def llm_list(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        store = get_store()
        tokens = _command_argv(event, "list", arg)
        text_only = "-t" in tokens
        kw = " ".join(t for t in tokens if t != "-t").strip()

        if text_only:
            lines = [_render_tree(store, event.unified_msg_origin)]
            if kw:
                # 简单过滤：保留含关键词的行 + api_base 标题行 + 汇总行
                lines = [
                    ln for ln in lines[0].split("\n")
                    if kw.lower() in ln.lower() or ln.startswith("🔗") or ln.startswith("——")
                ]
                if len(lines) <= 2:
                    lines = [f"未匹配到含 {kw!r} 的后端/实例/模型。", ""]
            yield event.plain_result("\n".join(lines))
            return

        # 默认：图片卡片渲染
        if render_catalog_image is None:
            yield event.plain_result(
                "Pillow 不可用，已回退纯文本。\n" + _render_tree(store, event.unified_msg_origin)
            )
            return
        try:
            img_path = render_catalog_image(store, event.unified_msg_origin)
        except ImportError:
            yield event.plain_result(
                "Pillow 不可用，已回退纯文本。\n" + _render_tree(store, event.unified_msg_origin)
            )
            return
        except Exception as e:
            logger.warning("渲染列表图片失败，回退纯文本：%s", e)
            yield event.plain_result(
                f"图片渲染失败（{e}），已回退纯文本：\n"
                + _render_tree(store, event.unified_msg_origin)
            )
            return

        try:
            yield event.image_result(img_path)
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass

    # ---------- /llmm search ----------

    @llmm.command("search")
    async def llm_search(self, event: AstrMessageEvent, arg: str = ""):
        """/llmm search <模型名关键词>  模糊搜索模型，多个关键词需全部命中。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        query = " ".join(_command_argv(event, "search", arg)).strip()
        if not query:
            yield event.plain_result(
                "用法：/llmm search <模型名关键词>\n"
                "例：/llmm search deepseek\n"
                "例：/llmm search qwen 72b（多个关键词需全部命中，忽略大小写）"
            )
            return

        store = get_store()
        items = store.search_models(query)
        if not items:
            yield event.plain_result(
                f"未找到模型名匹配 {query!r} 的模型。\n"
                "可缩短关键词重试，或用 /llmm list 查看已导入的模型。"
            )
            return

        routed = store.resolve(umo=event.unified_msg_origin)
        lines = [f"模型搜索 {query!r}：共 {len(items)} 条匹配"]
        for item in items:
            status = "（停用）" if item["disabled"] else ""
            if (
                not item["disabled"] and routed
                and item["instance_id"] == routed["instance"]["id"]
                and item["model"] == routed["model_name"]
            ):
                status += "  ● 使用中"
            lines.append(
                f"[{item['num']}] {item['model']}"
                f"{_modality_mark(item['modalities'])}{status}\n"
                f"  站点：{item['site']} · 分组：{item['instance_id']}"
            )
        lines.append("序号与 /llmm list 一致，可用 /llmm use <序号> 切换。")
        if render_catalog_image is None:
            yield event.plain_result(
                "Pillow 不可用，已回退纯文本。\n" + "\n".join(lines)
            )
            return
        try:
            img_path = render_catalog_image(
                store, event.unified_msg_origin, catalog=items,
            )
        except Exception as e:
            logger.warning("渲染搜索图片失败，回退纯文本：%s", e)
            yield event.plain_result("图片渲染失败，已回退纯文本。\n" + "\n".join(lines))
            return

        try:
            yield event.image_result(img_path)
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass

    # ---------- /llmm use ----------

    @llmm.command("use")
    async def llm_use(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "use", arg)
        conv_only = "-c" in tokens
        tokens = [t for t in tokens if t != "-c"]
        store = get_store()

        if not tokens:
            # 不带参数：显示当前状态
            umo = event.unified_msg_origin
            routed = store.resolve(umo=umo)
            override = store.get_conversation_override(umo)
            override_text = "（无）"
            if override:
                override_text = override["instance_id"]
                if override["model"]:
                    override_text += f"/{override['model']}"
            default_inst = store.data.get("default_instance", "")
            default_model = store.data.get("default_model", "")
            if routed:
                lines = [
                    f"当前生效：{routed['model_name']}（{routed['instance']['id']}）",
                    f"  会话 umo：{umo}",
                    f"  全局默认：{default_inst or '（未设置）'}"
                    + (f" / {default_model}" if default_model else " / 实例首个模型"),
                    f"  会话切换：{override_text}",
                    "",
                    "用法：/llmm use <序号|模型id|实例id> [-c]",
                    "  不带 -c：全局切换；带 -c：仅当前会话；use global 恢复全局默认。",
                ]
                if not default_inst and not override:
                    lines.append("")
                    lines.append(
                        "⚠️ 未设置全局默认，当前用的是兜底（第一个启用模型）。"
                        "建议 /llmm default <序号> 设定。"
                    )
                yield event.plain_result("\n".join(lines))
            else:
                yield event.plain_result("尚未配置模型。请先 /llmm import 或 /llmm add 添加后端。")
            return

        yield event.plain_result(_resolve_and_apply(store, event, tokens[0], conv_only))

    # ---------- /llmm default ----------

    @llmm.command("default")
    async def llm_default(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        target = " ".join(_command_argv(event, "default", arg)).strip()
        store = get_store()
        if not target:
            yield event.plain_result(
                f"当前全局默认：{store.data.get('default_instance') or '（未设置）'}"
                f" / {store.data.get('default_model') or '（实例默认模型）'}"
            )
            return

        item = store.set_default(target)
        if item is None:
            yield event.plain_result(f"未找到目标 {target!r}。")
            return
        yield event.plain_result(
            f"全局默认已设为 [{item['num']}] {item['model']}（{item['instance_id']}）。"
        )

    # ---------- /llmm fallback ----------

    @llmm.command("fallback")
    async def llm_fallback(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "fallback", arg)
        raw_action = tokens[0].casefold() if tokens else ""
        params = tokens[1:]
        aliases = {
            "status": "list", "show": "list",
            "enable": "on", "disable": "off",
            "del": "rm", "remove": "rm", "delete": "rm",
        }
        action = aliases.get(raw_action, raw_action) or "list"

        write_actions = {
            "add", "rm", "move", "clear", "on", "off", "mode", "max", "cooldown",
        }
        if action in write_actions and not self._is_bot_admin(event):
            yield event.plain_result(
                "此操作仅管理员可用。查看配置请用 /llmm fallback。"
            )
            return

        store = get_store()

        if action == "list":
            yield event.plain_result(
                _render_fallback(store, event.unified_msg_origin)
            )
            return

        if action == "add":
            if not params:
                yield event.plain_result(
                    "用法：/llmm fallback add <目录序号|实例/模型|模型名>...\n"
                    "例：/llmm fallback add 2 3\n"
                    "目标与 /llmm use 相同，按给出顺序追加到池尾。"
                )
                return

            result = store.add_fallback_targets(params)
            lines: list[str] = []
            if result["added"]:
                lines.append(f"已加入 {len(result['added'])} 条备选：")
                for offset, item in enumerate(result["added"]):
                    position = result["start_index"] + offset
                    suffix = "（已停用，运行时跳过）" if item["disabled"] else ""
                    lines.append(
                        f"  {position}. {item['instance_id']}/{item['model']}{suffix}"
                    )
            if result["duplicates"]:
                lines.append(f"跳过重复：{', '.join(result['duplicates'])}")
            if result["not_found"]:
                lines.append(f"未找到：{', '.join(result['not_found'])}")
            if result["overflow"]:
                lines.append(
                    f"备选池已满（最多 20 条），未加入：{', '.join(result['overflow'])}"
                )
            if not lines:
                lines.append("没有可加入的条目。")
            lines.append("")
            lines.append(_render_fallback(store, event.unified_msg_origin))
            if result["added"] and not result["enabled"]:
                lines.append("")
                lines.append("提示：故障转移尚未开启，执行 /llmm fallback on 生效。")
            yield event.plain_result("\n".join(lines))
            return

        if action == "rm":
            if not params:
                yield event.plain_result(
                    "用法：/llmm fallback rm <池序号...>\n"
                    "例：/llmm fallback rm 2\n"
                    "例：/llmm fallback rm 1-3\n"
                    "池序号以 /llmm fallback 显示为准，不是 /llmm list 的目录序号。"
                )
                return

            indexes, invalid = _parse_pool_indexes(params)
            lines = []
            if invalid:
                lines.append(f"无效池序号：{', '.join(invalid)}")
            if indexes:
                result = store.remove_fallback_indexes(indexes)
                if result["removed"]:
                    lines.append(f"已移除 {len(result['removed'])} 条：")
                    for item in result["removed"]:
                        lines.append(
                            f"  {item['index']}. "
                            f"{item['instance_id']}/{item['model']}"
                        )
                if result["invalid"]:
                    lines.append(
                        f"超出范围：{', '.join(str(i) for i in result['invalid'])}"
                    )
                if result["auto_disabled"]:
                    lines.append("备选池已空，故障转移已自动关闭。")
            if not lines:
                lines.append("没有移除任何条目。")
            lines.append("")
            lines.append(_render_fallback(store, event.unified_msg_origin))
            yield event.plain_result("\n".join(lines))
            return

        if action == "move":
            pool = store.fallback_config()["pool"]
            if len(params) != 2 or not all(p.isdigit() for p in params):
                yield event.plain_result(
                    "用法：/llmm fallback move <池序号> <新位置>\n"
                    f"例：/llmm fallback move 3 1（把第 3 条提到最前，当前 {len(pool)} 条）"
                )
                return
            index, new_index = int(params[0]), int(params[1])
            if not store.move_fallback(index, new_index):
                yield event.plain_result(
                    f"移动失败：池序号和新位置都需在 1..{len(pool)} 内。"
                )
                return
            yield event.plain_result(
                f"已把第 {index} 条移动到第 {new_index} 位。\n\n"
                + _render_fallback(store, event.unified_msg_origin)
            )
            return

        if action == "clear":
            count = len(store.fallback_config()["pool"])
            if not params or params[0].casefold() not in ("confirm", "确认"):
                yield event.plain_result(
                    f"这会清空 {count} 条备选并关闭故障转移。\n"
                    "确认请执行：/llmm fallback clear confirm"
                )
                return
            cleared = store.clear_fallback()
            yield event.plain_result(
                f"已清空 {cleared} 条备选，故障转移已关闭。"
                if cleared else "备选池本来就是空的。"
            )
            return

        if action in ("on", "off"):
            fb = store.fallback_config()
            if action == "on":
                if fb["enabled"]:
                    yield event.plain_result(
                        "故障转移已经是开启状态。\n\n"
                        + _render_fallback(store, event.unified_msg_origin)
                    )
                    return
                if not store.set_fallback_enabled(True):
                    yield event.plain_result(
                        "备选池为空，无法开启。先 /llmm fallback add <目录序号> 添加。"
                    )
                    return
                lines = ["故障转移已开启。"]
                if not store.resolve_fallback_entries():
                    lines.append(
                        "注意：池内没有可解析的条目，运行时不会发生转移。"
                        "用 /llmm fallback check 查看。"
                    )
            else:
                if not fb["enabled"]:
                    yield event.plain_result(
                        "当前已是关闭状态。\n\n"
                        + _render_fallback(store, event.unified_msg_origin)
                    )
                    return
                store.set_fallback_enabled(False)
                lines = ["故障转移已关闭，备选池保留。"]
            lines.append("")
            lines.append(_render_fallback(store, event.unified_msg_origin))
            yield event.plain_result("\n".join(lines))
            return

        if action == "mode":
            fb = store.fallback_config()
            if not params:
                yield event.plain_result(
                    f"当前模式：{fb['mode']}\n"
                    "  transient：每轮先试主模型，失败才临时降级（推荐）\n"
                    "  sticky：成功后当前会话粘滞，主模型冷却结束后自动恢复\n"
                    "用法：/llmm fallback mode transient|sticky"
                )
                return
            mode = params[0].casefold()
            if mode not in ("transient", "sticky"):
                yield event.plain_result("模式只能是 transient 或 sticky。")
                return
            store.set_fallback_mode(mode)
            if mode == "sticky":
                yield event.plain_result(
                    "已切换为 sticky：备选模型成功后，仅当前会话改用该模型；"
                    "主模型冷却结束后自动恢复。全局默认不变，"
                    "/llmm use global 可立即恢复。"
                )
            else:
                yield event.plain_result(
                    "已切换为 transient：每轮都会先尝试主模型，失败才临时降级，"
                    "不改变当前选择。"
                )
            return

        if action == "max":
            fb = store.fallback_config()
            if not params:
                yield event.plain_result(
                    f"当前尝试上限：{fb['max_attempts']} 次（含主模型）。\n"
                    "用法：/llmm fallback max 2-5"
                )
                return
            try:
                value = int(params[0])
            except ValueError:
                yield event.plain_result("上限必须是 2-5 的整数。")
                return
            if not store.set_fallback_max(value):
                yield event.plain_result("上限需在 2-5 之间（1 次等于关闭备选，请用 off）。")
                return
            yield event.plain_result(
                f"尝试上限已设为 {value} 次（含主模型，即最多再试 {value - 1} 个备选）。"
            )
            return

        if action == "cooldown":
            fb = store.fallback_config()
            if not params:
                yield event.plain_result(
                    f"当前冷却：{fb['cooldown_seconds']} 秒。\n"
                    "用法：/llmm fallback cooldown 0-3600（0 表示关闭冷却）"
                )
                return
            try:
                value = int(params[0])
            except ValueError:
                yield event.plain_result("冷却必须是 0-3600 的整数秒。")
                return
            if not store.set_fallback_cooldown(value):
                yield event.plain_result("冷却需在 0-3600 秒之间。")
                return
            yield event.plain_result(
                f"冷却已设为 {value} 秒。"
                + ("（已关闭冷却）" if value == 0 else "")
            )
            return

        if action == "check":
            statuses = store.fallback_status()
            if not statuses:
                yield event.plain_result(
                    "备选池为空。用 /llmm fallback add <目录序号> 添加。"
                )
                return
            lines = ["备选池检查（未调用模型）："]
            ok = disabled = missing = 0
            for item in statuses:
                num = f" [{item['num']}]" if item["num"] is not None else ""
                if item["status"] == "missing":
                    missing += 1
                    state = "失效：实例或模型已删除"
                elif item["status"] == "disabled":
                    disabled += 1
                    state = "停用：运行时跳过"
                else:
                    ok += 1
                    state = "可解析" + _modality_mark(item["modalities"])
                    if item["cooldown"] > 0:
                        state += f"，冷却中 {item['cooldown']}s"
                lines.append(
                    f"  {item['index']}. {item['instance_id']}/{item['model']}"
                    f"{num}  {state}"
                )
            lines.append(
                f"—— 可解析 {ok} 条 / 停用 {disabled} 条 / 失效 {missing} 条 ——"
            )
            yield event.plain_result("\n".join(lines))
            return

        yield event.plain_result(_fallback_usage())

    # ---------- /llmm add ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("add")
    async def llm_add(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "add", arg)
        if len(tokens) < 3:
            yield event.plain_result(
                "用法：/llmm add <站名> <base_url> <key> [类型]\n"
                "例：/llmm add DeepSeek https://api.deepseek.com/v1 sk-xxx\n"
                "类型默认 openai_chat_completion（Anthropic/Gemini 等可显式指定）。"
            )
            return

        site, api_base, key = tokens[0], tokens[1], tokens[2]
        stype = tokens[3] if len(tokens) > 3 else "openai_chat_completion"
        store = get_store()
        try:
            src = store.add_source(site, api_base, key, stype=stype)
        except ValueError as e:
            yield event.plain_result(f"添加失败：{e}")
            return

        yield event.plain_result(
            f"已添加后端源：{src['site']}（id={src['id']}）\n"
            f"  Base URL：{src['api_base']}\n"
            "下一步：/llmm group add <站名> <分组名> [模型...] 建立分组。"
        )

    # ---------- /llmm import ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("import")
    async def llm_import(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        store = get_store()
        plugin_data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        # 候选路径：AstrBot 数据目录下的 cmd_config.json
        candidates = [
            plugin_data_dir.parent.parent / "cmd_config.json",
            Path("/AstrBot/data/cmd_config.json"),
            Path.home() / "data" / "cmd_config.json",
        ]
        config_path = None
        for c in candidates:
            if c.exists():
                config_path = c
                break

        if config_path is None:
            yield event.plain_result(
                "未找到 cmd_config.json。已尝试路径：\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\n请确认 AstrBot 数据目录结构，或手动用 /llmm add 添加。"
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
        except Exception as e:
            yield event.plain_result(f"读取 cmd_config.json 失败：{e}")
            return

        result = store.import_from_astrbot_config(config)
        lines = [
            f"已从系统配置导入（{config_path.name}）：",
            f"  新增源：{result['imported_sources']} 个",
            f"  新增分组：{result['imported_instances']} 个",
            f"  新增模型：{result['imported_models']} 个",
            f"  当前总计：{result['total_sources']} 个源 / {result['total_models']} 个模型",
        ]
        if result["skipped"]:
            lines.append(f"  跳过（非聊天模型/无 provider_source_id）：{', '.join(result['skipped'])}")

        # 自动保留原默认模型的提示
        if result.get("auto_default_set"):
            lines.append(f"  ✅ 已自动保留原默认模型：{result['original_default']}")
        elif result.get("original_default"):
            lines.append(
                f"  ⚠️ 原默认模型 {result['original_default']} 未在目录中匹配到，"
                "请用 /llmm default <序号> 手动设置。"
            )
        else:
            lines.append(
                "  ⚠️ 未检测到原默认模型（default_provider_id / agent_runner 均为空），"
                "当前用兜底模型，建议 /llmm default <序号> 设置。"
            )

        lines.append("\n用 /llmm list 查看，/llmm use <序号> 切换。")
        yield event.plain_result("\n".join(lines))

    # ---------- /llmm resync ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("resync")
    async def llm_resync(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        store = get_store()
        plugin_data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        candidates = [
            plugin_data_dir.parent.parent / "cmd_config.json",
            Path("/AstrBot/data/cmd_config.json"),
            Path.home() / "data" / "cmd_config.json",
        ]
        config_path = None
        for c in candidates:
            if c.exists():
                config_path = c
                break

        if config_path is None:
            yield event.plain_result(
                "未找到 cmd_config.json。已尝试路径：\n"
                + "\n".join(f"  {c}" for c in candidates)
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
        except Exception as e:
            yield event.plain_result(f"读取 cmd_config.json 失败：{e}")
            return

        result = store.resync_from_astrbot_config(config)
        lines = [
            f"已从系统配置完全重建（{config_path.name}）：",
            f"  重建前：{result['old_sources']} 个源 / {result['old_models']} 个模型",
            f"  重建后：{result['new_sources']} 个源 / {result['new_models']} 个模型",
        ]
        if result["default_preserved"]:
            lines.append("  ✅ 全局默认模型已保留")
        else:
            lines.append(
                "  ⚠️ 原全局默认模型已不存在，已清空（用 /llmm default <序号> 重新设置）"
            )
        if result["overrides_cleaned"]:
            lines.append(f"  已清理 {result['overrides_cleaned']} 个无效会话切换")
        if result["skipped"]:
            lines.append(f"  跳过（非聊天模型）：{', '.join(result['skipped'])}")
        if result.get("backup_file"):
            lines.append(f"  原配置已备份为：{result['backup_file']}")
        lines.append("\n用 /llmm list 查看最新配置。")
        yield event.plain_result("\n".join(lines))

    # ---------- /llmm group ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("group")
    async def llm_group(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "group", arg)
        store = get_store()
        if not tokens:
            yield event.plain_result(
                "子命令：add / rm / enable / disable\n"
                "例：/llmm group add DeepSeek 推理 deepseek-reasoner"
            )
            return

        sub = tokens[0].lower()

        if sub == "add":
            if len(tokens) < 3:
                yield event.plain_result("用法：/llmm group add <站名> <分组名> [模型...]")
                return
            try:
                inst = store.add_instance(tokens[1], tokens[2], tokens[3:])
            except ValueError as e:
                yield event.plain_result(f"添加失败：{e}")
                return
            added = len(inst.get("models", []))
            extra = f"，已挂载 {added} 个模型" if added else ""
            yield event.plain_result(
                f"已创建分组实例：{inst['id']}{extra}\n"
                "后续可用 /llmm model add <实例id> <模型id>... 继续挂载模型。"
            )
            return

        if sub == "rm":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llmm group rm <实例id>")
                return
            ok = store.remove_instance(tokens[1])
            yield event.plain_result(f"已删除实例 {tokens[1]}。" if ok else f"未找到实例 {tokens[1]}。")
            return

        if sub in ("enable", "disable"):
            if len(tokens) < 2:
                yield event.plain_result(f"用法：/llmm group {sub} <实例id>")
                return
            ok = store.set_instance_enabled(tokens[1], sub == "enable")
            yield event.plain_result(
                f"实例 {tokens[1]} 已{'启用' if sub == 'enable' else '停用'}。"
                if ok else f"未找到实例 {tokens[1]}。"
            )
            return

        yield event.plain_result(f"未知子命令 {sub!r}。支持：add / rm / enable / disable")

    # ---------- /llmm model ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("model")
    async def llm_model(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "model", arg)
        store = get_store()
        if not tokens:
            yield event.plain_result(
                "子命令：add / rm\n例：/llmm model add deepseek_推理 deepseek-reasoner"
            )
            return

        sub = tokens[0].lower()
        if len(tokens) < 3:
            yield event.plain_result(f"用法：/llmm model {sub} <实例id> <模型id>...")
            return

        inst_id, models = tokens[1], tokens[2:]
        try:
            if sub == "add":
                n = store.add_models(inst_id, models)
                yield event.plain_result(
                    f"实例 {inst_id} 新增 {n} 个模型：{' '.join(models)}。" if n
                    else f"模型已全部存在或未新增（实例 {inst_id}）。"
                )
            elif sub == "rm":
                n = store.remove_models(inst_id, models)
                yield event.plain_result(
                    f"实例 {inst_id} 移除 {n} 个模型。" if n
                    else f"未找到可移除的模型（实例 {inst_id}）。"
                )
            else:
                yield event.plain_result(f"未知子命令 {sub!r}。支持：add / rm")
        except ValueError as e:
            yield event.plain_result(f"操作失败：{e}")

    # ---------- /llmm enable / disable ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("enable")
    async def llm_enable(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        target = " ".join(_command_argv(event, "enable", arg)).strip()
        if not target:
            yield event.plain_result("用法：/llmm enable <实例id>")
            return
        store = get_store()
        ok = store.set_instance_enabled(target, True)
        yield event.plain_result(f"已启用 {target}。" if ok else f"未找到实例 {target}。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("disable")
    async def llm_disable(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        target = " ".join(_command_argv(event, "disable", arg)).strip()
        if not target:
            yield event.plain_result("用法：/llmm disable <实例id>")
            return
        store = get_store()
        ok = store.set_instance_enabled(target, False)
        yield event.plain_result(f"已停用 {target}。" if ok else f"未找到实例 {target}。")

    # ---------- /llmm rm ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llmm.command("rm")
    async def llm_rm(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        tokens = _command_argv(event, "rm", arg)
        if not tokens:
            yield event.plain_result(
                "用法：\n"
                "  /llmm rm source <站名|源id>   删除一级源（含所有分组和模型）\n"
                "  /llmm rm instance <实例id>     删除实例\n"
                "  /llmm rm model <实例id> <模型id>  删除模型\n"
                "注意：删除后立即生效，用 /llmm list 查看。"
            )
            return

        sub = tokens[0].lower()
        store = get_store()

        if sub == "source":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llmm rm source <站名|源id>")
                return
            target = tokens[1]
            ok = store.remove_source(target)
            if ok:
                yield event.plain_result(
                    f"已删除一级源 {target}（含其下所有分组和模型）。\n"
                    "用 /llmm list 查看最新配置。"
                )
            else:
                yield event.plain_result(f"未找到一级源 {target}（用 /llmm list 查看站名/源id）。")
            return

        if sub == "instance":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llmm rm instance <实例id>")
                return
            ok = store.remove_instance(tokens[1])
            yield event.plain_result(f"已删除实例 {tokens[1]}。" if ok else f"未找到实例 {tokens[1]}。")
            return

        if sub == "model":
            if len(tokens) < 3:
                yield event.plain_result("用法：/llmm rm model <实例id> <模型id>")
                return
            n = store.remove_models(tokens[1], tokens[2:])
            yield event.plain_result(
                f"实例 {tokens[1]} 移除 {n} 个模型。" if n
                else f"未找到可移除的模型（实例 {tokens[1]}）。"
            )
            return

        yield event.plain_result(f"未知子命令 {sub!r}。支持：source / instance / model")

    # ---------- /llmm stats ----------

    @llmm.command("stats")
    async def llm_stats(self, event: AstrMessageEvent, arg: str = ""):
        """查看各模型 Token 消耗排行。

        参数：
          （无）  默认最近 7 天
          1d     最近 1 天
          7d     最近 7 天
          30d    最近 30 天
          all    全部历史（受 30 天保留期限制）
        """
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        store = get_store()

        # 解析时间范围参数
        range_tokens = _command_argv(event, "stats", arg)
        arg_lower = (range_tokens[0] if range_tokens else "").lower()
        days_map = {
            "1d": 1, "1": 1, "day": 1, "today": 1,
            "7d": 7, "7": 7, "week": 7,
            "30d": 30, "30": 30, "month": 30,
            "all": None, "total": None, "全部": None,
        }
        days = days_map.get(arg_lower, 7)  # 默认 7 天

        range_text = "全部历史" if days is None else f"最近 {days} 天"
        items = store.stats.aggregate(days)

        if not items:
            yield event.plain_result(
                f"📊 {range_text} 暂无 Token 消耗记录。\n"
                "（统计功能刚启用，需要一些对话后才会有数据；"
                "流式对话在提供商返回 usage 时才会记录）"
            )
            return

        total_calls = sum(i["calls"] for i in items)
        total_tokens = sum(i["total_tokens"] for i in items)
        fallback_calls = store.stats.fallback_calls(days)

        lines = [
            f"📊 {range_text} 模型 Token 消耗排行",
            "━" * 30,
        ]

        for idx, item in enumerate(items, 1):
            model = item["model"]
            instance = item["instance"]
            calls = item["calls"]
            prompt = item["prompt_tokens"]
            completion = item["completion_tokens"]
            total = item["total_tokens"]

            lines.append(f"{idx}. {model}（{instance}）")
            lines.append(f"   总 Token: {total:,} | 调用: {calls} 次")
            lines.append(f"   输入: {prompt:,} | 输出: {completion:,}")
            if idx < len(items):
                lines.append("")

        lines.append("━" * 30)
        lines.append(f"合计: {total_tokens:,} Token | {total_calls} 次调用")
        if fallback_calls:
            lines.append(f"其中降级响应: {fallback_calls} 次")
        lines.append("")
        lines.append("提示: /llmm stats 1d | 7d | 30d | all  切换时间范围")
        lines.append("（流式对话需提供商返回 usage 才会记录）")

        yield event.plain_result("\n".join(lines))

    # ---------- /llmm test ----------

    async def _test_one_model(self, store: Store, item: dict) -> tuple:
        """测试单个模型连通性与延迟。
        返回 (item, success: bool, latency_ms: float, error_msg: str | None)
        所有异常均内部捕获，确保不会向上抛出影响批量测试。
        """
        source = store.find_source(item["source_id"])
        inst = store.find_instance(item["instance_id"])
        if source is None or inst is None or item.get("disabled", False):
            return (item, False, 0.0, "源或实例被停用")

        try:
            from .manager_provider import LLMManagerProvider

            provider = LLMManagerProvider({"model": item["model"], "type": "llm_manager"}, {})
            backend = provider._backend_for(source)
            provider._apply_model_to_backend(backend, item["model"])

            start = time.time()
            await asyncio.wait_for(backend.test(), timeout=60)
            cost = (time.time() - start) * 1000
            return (item, True, cost, None)
        except Exception as e:
            return (item, False, 0.0, str(e))

    @llmm.command("test")
    async def llm_test(
        self,
        event: AstrMessageEvent,
        arg: str = "",
        _p1: str = "",
        _p2: str = "",
        _p3: str = "",
        _p4: str = "",
        _p5: str = "",
        _p6: str = "",
        _p7: str = "",
        _p8: str = "",
        _p9: str = "",
    ):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        store = get_store()
        # 兼容 AstrBot 两种参数传递方式：
        # 方式一：arg 包含全部文本（"30 31 36"），其余 _p* 为空
        # 方式二：arg 只含第一个（"30"），其余拆分到 _p1="31", _p2="36"
        fallback = " ".join(
            part for part in (arg, _p1, _p2, _p3, _p4, _p5, _p6, _p7, _p8, _p9) if part
        )
        tokens = _command_argv(event, "test", fallback)
        logger.debug("[LLM Manager] /llmm test 参数解析 | tokens=%r", tokens)

        if len(tokens) > 10:
            yield event.plain_result(
                f"一次最多测试 10 个模型（当前 {len(tokens)} 个），请分批执行。"
            )
            return

        # 不带参数：测试当前正在使用的模型
        if not tokens:
            routed = store.resolve(umo=event.unified_msg_origin)
            if routed is None:
                yield event.plain_result("尚未配置任何可用模型。")
                return
            # 从目录中找到当前模型的真实序号
            cur_num = "?"
            for c in store.build_catalog():
                if c["instance_id"] == routed["instance"]["id"] and c["model"] == routed["model_name"]:
                    cur_num = c["num"]
                    break
            item = {
                "num": cur_num,
                "model": routed["model_name"],
                "instance_id": routed["instance"]["id"],
                "source_id": routed["source"]["id"],
                "modalities": routed["model_entry"].get("modalities", []),
                "disabled": False,
            }
            _, success, latency, error = await self._test_one_model(store, item)
            if success:
                yield event.plain_result(
                    f"当前使用：[{cur_num}] {item['model']}（{item['instance_id']}）\n"
                    f"✅ 连通正常，耗时 {latency:.0f} ms"
                )
            else:
                yield event.plain_result(
                    f"当前使用：[{cur_num}] {item['model']}（{item['instance_id']}）\n"
                    f"❌ 测试失败：{error}"
                )
            return

        # 带一个或多个参数：逐个解析并并行测试
        items = []
        not_found = []
        for target in tokens:
            item = store.find_in_catalog(target)
            if item is None:
                not_found.append(target)
            else:
                items.append(item)

        if not items:
            yield event.plain_result(
                f"未找到任何目标：{', '.join(not_found)}\n"
                "用法：/llmm test <序号|模型id|实例id>...（/llmm list 查看序号）"
            )
            return

        # 并行测试所有模型（return_exceptions 确保单个异常不中断整体）
        raw_results = await asyncio.gather(
            *[self._test_one_model(store, item) for item in items],
            return_exceptions=True,
        )
        # 归一化结果：gather 返回的异常对象转成 (item, False, 0, error_msg)
        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                results.append((items[i], False, 0.0, str(r)))
            else:
                results.append(r)

        # 汇总输出
        lines = []
        success_count = 0
        for item, success, latency, error in results:
            if success:
                success_count += 1
                lines.append(f"✅ [{item['num']}] {item['model']}（{item['instance_id']}） {latency:.0f} ms")
            else:
                lines.append(f"❌ [{item['num']}] {item['model']}（{item['instance_id']}） 失败：{error}")
        if not_found:
            lines.append(f"⚠️ 未找到：{', '.join(not_found)}")
        lines.append(f"—— {success_count}/{len(results)} 成功 ——")
        yield event.plain_result("\n".join(lines))

    # ---------- /llmm help ----------

    @llmm.command("help")
    async def llm_help(self, event: AstrMessageEvent, arg: str = ""):
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return

        help_text = (
            "LLM 供应商管理插件\n"
            "三级结构：API Base URL → 实例(站名_分组名) → 模型[全局序号]\n\n"
            "查看/切换：\n"
            "  /llmm list [关键词] [-t]      三级卡片图片查看；-t 纯文本\n"
            "  /llmm search <模型名关键词>   模糊搜索并返回卡片图片；多关键词全部匹配\n"
            "  /llmm use <序号|模型id|实例id> [-c]   全局切换；-c 仅当前会话\n"
            "  /llmm use global            清除会话切换，回到全局默认\n"
            "  /llmm default <目标>        设置全局默认模型/实例\n"
            "故障转移（管理员配置，查看跟随 admin_only）：\n"
            "  /llmm fallback              查看开关、模式与备选池\n"
            "  /llmm fallback add <目标...>   添加备选（目录序号/实例/模型）\n"
            "  /llmm fallback rm <池序号...>  删除，支持 1-3 区间\n"
            "  /llmm fallback move <序号> <新位置>   调整优先级\n"
            "  /llmm fallback clear confirm   清空并关闭\n"
            "  /llmm fallback on|off       启用/停用\n"
            "  /llmm fallback mode transient|sticky\n"
            "  /llmm fallback max 2-5      总尝试次数（含主模型）\n"
            "  /llmm fallback cooldown 0-3600   冷却秒数\n"
            "  /llmm fallback check        静态校验，不调用模型\n"
            "管理（管理员）：\n"
            "  /llmm import                 从系统配置(cmd_config.json)导入已有供应商\n"
            "  /llmm add <站名> <base_url> <key> [类型]   新增后端源\n"
            "  /llmm group add <站名> <分组名> [模型...]   新增分组实例\n"
            "  /llmm group rm|enable|disable <实例id>\n"
            "  /llmm model add|rm <实例id> <模型id>...    挂载/移除模型\n"
            "运维：\n"
            "  /llmm stats [1d|7d|30d|all]  各模型 Token 消耗排行\n"
            "  /llmm test <目标>           连通性与时延测试\n"
            "配置存放：data/plugin_data/astrbot_plugin_llm_manager/backends.json"
        )
        yield event.plain_result(help_text)
