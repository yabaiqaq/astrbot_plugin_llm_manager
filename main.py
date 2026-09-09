"""LLM 供应商统一管理插件主入口。

设计：AstrBot 中只注册一个虚拟 Provider（LLM Manager），全部真实后端、
分组与模型由插件自己的 backends.json 管理；指令层提供三级查看与全局切换。
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from pathlib import Path

from astrbot import logger
from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

from ._deps import ensure_pillow
from .manager_provider import (
    LLMManagerProvider,
    PROVIDER_TYPE_NAME,
    get_store,
    set_store,
)
from .register_provider import (
    register_or_replace_provider_adapter,
    unregister_provider_adapter,
)

# 运行时自动检测并安装 Pillow（在导入 renderer 之前执行）
ensure_pillow()

from .renderer import render_catalog_image  # noqa: E402
from .store import DEFAULT_MODALITIES, Store

PLUGIN_NAME = "astrbot_plugin_llm_manager"
PROVIDER_TEMPLATE_KEY = "LLM Manager"

# ---------------------------------------------------------------
# Provider 注册（模块导入即注册，与社区 Provider 插件一致）
# ---------------------------------------------------------------


def _provider_template() -> dict:
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
    }


def _inject_provider_source_template(template_key: str, template: dict) -> None:
    try:
        from astrbot.core.config.default import CONFIG_METADATA_2

        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template[template_key] = copy.deepcopy(template)
    except Exception as e:  # noqa: BLE001
        logger.warning("注入 WebUI 提供商模板失败（不影响功能）: %s", e)


def _remove_provider_source_template(template_key: str) -> None:
    try:
        from astrbot.core.config.default import CONFIG_METADATA_2

        config_template = CONFIG_METADATA_2["provider_group"]["metadata"]["provider"][
            "config_template"
        ]
        config_template.pop(template_key, None)
    except Exception:  # noqa: BLE001
        return


def _register_provider() -> None:
    template = _provider_template()
    register_or_replace_provider_adapter(
        provider_type_name=PROVIDER_TYPE_NAME,
        desc="LLM 供应商统一管理（路由/委托）",
        cls_type=LLMManagerProvider,
        default_config_tmpl=template,
        provider_display_name=PROVIDER_TEMPLATE_KEY,
    )
    _inject_provider_source_template(PROVIDER_TEMPLATE_KEY, template)


_register_provider()

# ---------------------------------------------------------------
# 指令组（定义在类内，见下方 LLMManagerPlugin.llm）
# ---------------------------------------------------------------


def _modality_mark(modalities: list[str]) -> str:
    marks = []
    if "tool_use" in (modalities or DEFAULT_MODALITIES):
        marks.append("工具")
    if "image" in (modalities or []):
        marks.append("视觉")
    if "audio" in (modalities or []):
        marks.append("语音")
    return ("·" + "/".join(marks)) if marks else ""


def _render_tree(store: Store, umo: str | None = None) -> str:
    """四级树形视图：Base URL -> 源(site) -> 实例(站名_分组名) -> 模型[序号]。
    存储层每个 provider_source 独立（保留各自 key），显示层按 api_base 合并。"""
    lines: list[str] = []
    override_id = store.get_conversation_override(umo) if umo else None
    default_inst = store.data.get("default_instance", "")
    default_model = store.data.get("default_model", "")

    # 解析当前真正生效的模型（全局默认 或 会话 override）
    current_routed = store.resolve(umo=umo)
    cur_instance_id = current_routed["instance"]["id"] if current_routed else ""
    cur_model_name = current_routed["model_name"] if current_routed else ""

    # 按 api_base 分组（保持原有顺序）
    api_groups: dict[str, list[dict]] = {}
    for src in store.sources():
        api_base = str(src.get("api_base", ""))
        api_groups.setdefault(api_base, []).append(src)

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
                if inst_id == override_id:
                    marks.append("← 本会话")
                mark_txt = f"  [{' '.join(marks)}]" if marks else ""
                lines.append(f"    └─ {inst_id}{mark_txt}")
                models = inst.get("models", [])
                if not models:
                    lines.append("        （暂无模型，用 /llm model add 挂载）")
                    continue
                for item in store.build_catalog():
                    if item["instance_id"] != inst_id:
                        continue
                    dis = "（停用）" if item["disabled"] else ""
                    cur = ""
                    if (
                        not dis
                        and inst_id == cur_instance_id
                        and item["model"] == cur_model_name
                    ):
                        cur = "  ● 使用中"
                    lines.append(
                        f"        ├─ [{item['num']}] {item['model']}"
                        f"{_modality_mark(item['modalities'])}{dis}{cur}"
                    )

    if not store.sources():
        lines.append("（尚未配置任何后端。示例：/llm add DeepSeek https://api.deepseek.com/v1 sk-xxx）")
    total = len(store.build_catalog())
    lines.append(f"—— 共 {len(store.sources())} 个源 / {total} 个模型 ——")
    return "\n".join(lines)


def _resolve_and_apply(
    store: Store, event: AstrMessageEvent, target: str, conv_only: bool
) -> str:
    if target == "global":
        umo = event.unified_msg_origin
        cleared = store.clear_conversation_override(umo)
        cur = store.data.get("default_instance", "")
        return (
            f"已清除本会话切换，回到全局默认：{cur or '（未设置）'}。"
            if cleared
            else f"本会话本就没有独立切换，全局默认：{cur or '（未设置）'}。"
        )

    item = store.find_in_catalog(target)
    if item is None:
        return (
            f"未找到目标 {target!r}。可用：模型序号（/llm list 查看）、模型 ID、"
            "实例 ID（站名_分组名）。"
        )

    umo = event.unified_msg_origin
    if conv_only:
        store.set_conversation_override(umo, target)
        return f"本会话已切换到 [{item['num']}] {item['model']}（{item['instance_id']}）。"
    store.set_default(target)
    return f"已全局切换到 [{item['num']}] {item['model']}（{item['instance_id']}）。"

# ---------------------------------------------------------------
# Star 插件
# ---------------------------------------------------------------


@register(PLUGIN_NAME, "yabaiqaq", "LLM 供应商三级管理插件：统一后端、分组与模型切换", "0.1.0")
class LLMManagerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        store = Store(data_dir / "backends.json")
        set_store(store)
        logger.info(
            "LLM Manager 插件已加载，配置仓库：%s", data_dir / "backends.json"
        )

    async def terminate(self) -> None:
        unregister_provider_adapter(PROVIDER_TYPE_NAME)
        _remove_provider_source_template(PROVIDER_TEMPLATE_KEY)

    # ---------- 权限检查 ----------

    def _is_bot_admin(self, event: AstrMessageEvent) -> bool:
        """与框架级 PermissionType.ADMIN 同源的管理员判断：查全局配置 admins_id。
        event.is_admin() 在群聊中可能未被正确设置 role，故兜底直接查 admins_id。"""
        # 1. 优先 event.is_admin()
        try:
            attr = getattr(event, "is_admin", None)
            if callable(attr):
                if attr():
                    return True
            elif attr:
                return True
        except Exception:
            pass

        # 2. 兜底：直接查全局配置 admins_id（框架 PermissionType.ADMIN 同源）
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
        """所有 /llm 指令的统一权限检查。配置 admin_only=true（默认）时仅管理员可用。"""
        if not self.config.get("admin_only", True):
            return True
        return self._is_bot_admin(event)

    def _deny_if_not_admin(self, event: AstrMessageEvent):
        """非管理员时返回拒绝消息，供各指令开头调用。返回 None 表示放行。"""
        if not self._is_allowed(event):
            return event.plain_result("/llm 指令仅管理员可用。如需放开，请在插件配置中关闭 admin_only。")
        return None

    # ---------- 指令组（必须先定义，子命令挂在它下面） ----------

    @filter.command_group("llm")
    def llm(self):
        """LLM 供应商管理指令组。"""
        pass

    # ---------- 三级查看 ----------

    @llm.command("list")
    async def llm_list(self, event: AstrMessageEvent, arg: str = ""):
        """/llm list [关键词] [-t]  三级卡片图片查看；-t 纯文本。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        store = get_store()
        tokens = arg.split()
        text_only = "-t" in tokens
        kw = " ".join(t for t in tokens if t != "-t").strip()

        # 纯文本模式
        if text_only:
            lines = [_render_tree(store, event.unified_msg_origin)]
            if kw:
                lines = [
                    ln for ln in lines[0].split("\n")
                    if kw.lower() in ln.lower() or ln.startswith("🔗") or ln.startswith("——")
                ]
                if len(lines) <= 2:
                    lines = [f"未匹配到含 {kw!r} 的后端/实例/模型。", ""]
            yield event.plain_result("\n".join(lines))
            return

        # 图片模式（默认）
        try:
            img_path = render_catalog_image(store, event.unified_msg_origin)
        except ImportError:
            yield event.plain_result(
                "Pillow 不可用，已回退纯文本。\n" + _render_tree(store, event.unified_msg_origin)
            )
            return
        except Exception as e:  # noqa: BLE001
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

    # ---------- 全局/会话切换 ----------

    @llm.command("use")
    async def llm_use(self, event: AstrMessageEvent, arg: str = ""):
        """/llm use <序号|模型id|实例id> [-c]  全局切换；-c 仅当前会话。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        tokens = arg.split()
        conv_only = "-c" in tokens
        tokens = [t for t in tokens if t != "-c"]
        store = get_store()
        if not tokens:
            umo = event.unified_msg_origin
            routed = store.resolve(umo=umo)
            override_id = store.get_conversation_override(umo)
            default_inst = store.data.get("default_instance", "")
            default_model = store.data.get("default_model", "")
            if routed:
                lines = [
                    f"当前生效：{routed['model_name']}（{routed['instance']['id']}）",
                    f"  会话 umo：{umo}",
                    f"  全局默认：{default_inst or '（未设置）'}"
                    + (f" / {default_model}" if default_model else " / 实例首个模型"),
                    f"  会话切换：{override_id or '（无）'}",
                    "",
                    "用法：/llm use <序号|模型id|实例id> [-c]",
                    "  不带 -c：全局切换；带 -c：仅当前会话；use global 恢复全局默认。",
                ]
                if not default_inst and not override_id:
                    lines.append("")
                    lines.append("⚠️ 未设置全局默认，当前用的是兜底（第一个启用模型）。建议 /llm default <序号> 设定。")
                yield event.plain_result("\n".join(lines))
            else:
                yield event.plain_result("尚未配置模型。请先 /llm import 或 /llm add 添加后端。")
            return
        yield event.plain_result(_resolve_and_apply(store, event, tokens[0], conv_only))

    @llm.command("default")
    async def llm_default(self, event: AstrMessageEvent, arg: str = ""):
        """/llm default <序号|模型id|实例id>  设置全局默认。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        target = arg.strip()
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

    # ---------- 后端管理（管理员） ----------

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("add")
    async def llm_add(self, event: AstrMessageEvent, arg: str = ""):
        """/llm add <站名> <base_url> <key> [类型]  新增一级后端源。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        tokens = arg.split()
        if len(tokens) < 3:
            yield event.plain_result(
                "用法：/llm add <站名> <base_url> <key> [类型]\n"
                "例：/llm add DeepSeek https://api.deepseek.com/v1 sk-xxx\n"
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
            "下一步：/llm group add <站名> <分组名> [模型...] 建立分组。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("import")
    async def llm_import(self, event: AstrMessageEvent, arg: str = ""):
        """/llm import  从 AstrBot 系统配置(cmd_config.json)导入已有供应商和模型。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        store = get_store()
        # 插件数据目录在 <data_dir>/plugin_data/<plugin_name>/，cmd_config.json 在 <data_dir>/
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
                + "\n请确认 AstrBot 数据目录结构，或手动用 /llm add 添加。"
            )
            return
        try:
            with open(config_path, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
        except Exception as e:  # noqa: BLE001
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
        if result.get("auto_default_set"):
            lines.append(f"  ✅ 已自动保留原默认模型：{result['original_default']}")
        elif result.get("original_default"):
            lines.append(
                f"  ⚠️ 原默认模型 {result['original_default']} 未在目录中匹配到，"
                "请用 /llm default <序号> 手动设置。"
            )
        else:
            lines.append(
                "  ⚠️ 未检测到原默认模型（default_provider_id 为空），"
                "当前用兜底模型，建议 /llm default <序号> 设置。"
            )
        lines.append("\n用 /llm list 查看，/llm use <序号> 切换。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("resync")
    async def llm_resync(self, event: AstrMessageEvent, arg: str = ""):
        """/llm resync  从系统配置(cmd_config.json)完全重建，解决手动删除供应商后不同步的问题。"""
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
        except Exception as e:  # noqa: BLE001
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
            lines.append("  ⚠️ 原全局默认模型已不存在，已清空（用 /llm default <序号> 重新设置）")
        if result["overrides_cleaned"]:
            lines.append(f"  已清理 {result['overrides_cleaned']} 个无效会话切换")
        if result["skipped"]:
            lines.append(f"  跳过（非聊天模型）：{', '.join(result['skipped'])}")
        lines.append("\n用 /llm list 查看最新配置。")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("group")
    async def llm_group(self, event: AstrMessageEvent, arg: str = ""):
        """/llm group add <站名> <分组名> [模型...]  新增二级实例（id=站名_分组名）。
        /llm group rm <实例id>  删除实例。
        /llm group enable|disable <实例id>  启停实例。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        tokens = arg.split()
        store = get_store()
        if not tokens:
            yield event.plain_result(
                "子命令：add / rm / enable / disable\n"
                "例：/llm group add DeepSeek 推理 deepseek-reasoner"
            )
            return
        sub = tokens[0].lower()
        if sub == "add":
            if len(tokens) < 3:
                yield event.plain_result(
                    "用法：/llm group add <站名> <分组名> [模型...]"
                )
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
                "后续可用 /llm model add <实例id> <模型id>... 继续挂载模型。"
            )
            return
        if sub == "rm":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llm group rm <实例id>")
                return
            ok = store.remove_instance(tokens[1])
            yield event.plain_result(
                f"已删除实例 {tokens[1]}。" if ok else f"未找到实例 {tokens[1]}。"
            )
            return
        if sub in ("enable", "disable"):
            if len(tokens) < 2:
                yield event.plain_result(f"用法：/llm group {sub} <实例id>")
                return
            ok = store.set_instance_enabled(tokens[1], sub == "enable")
            yield event.plain_result(
                f"实例 {tokens[1]} 已{'启用' if sub == 'enable' else '停用'}。"
                if ok
                else f"未找到实例 {tokens[1]}。"
            )
            return
        yield event.plain_result(f"未知子命令 {sub!r}。支持：add / rm / enable / disable")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("model")
    async def llm_model(self, event: AstrMessageEvent, arg: str = ""):
        """/llm model add <实例id> <模型id>...  挂载模型。
        /llm model rm <实例id> <模型id>...  移除模型。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        tokens = arg.split()
        store = get_store()
        if not tokens:
            yield event.plain_result("子命令：add / rm\n例：/llm model add deepseek_推理 deepseek-reasoner")
            return
        sub = tokens[0].lower()
        if len(tokens) < 3:
            yield event.plain_result(f"用法：/llm model {sub} <实例id> <模型id>...")
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

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("enable")
    async def llm_enable(self, event: AstrMessageEvent, arg: str = ""):
        """/llm enable <实例id>  启用实例。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        store = get_store()
        ok = store.set_instance_enabled(arg.strip(), True)
        yield event.plain_result(
            f"已启用 {arg.strip()}。" if ok else f"未找到实例 {arg.strip()}。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("disable")
    async def llm_disable(self, event: AstrMessageEvent, arg: str = ""):
        """/llm disable <实例id>  停用实例。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        store = get_store()
        ok = store.set_instance_enabled(arg.strip(), False)
        yield event.plain_result(
            f"已停用 {arg.strip()}。" if ok else f"未找到实例 {arg.strip()}。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @llm.command("rm")
    async def llm_rm(self, event: AstrMessageEvent, arg: str = ""):
        """/llm rm source <站名|源id>  删除一级源（含所有分组和模型）
        /llm rm instance <实例id>  删除实例（同 /llm group rm）
        /llm rm model <实例id> <模型id>  删除模型（同 /llm model rm）"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        tokens = arg.split()
        if not tokens:
            yield event.plain_result(
                "用法：\n"
                "  /llm rm source <站名|源id>   删除一级源（含所有分组和模型）\n"
                "  /llm rm instance <实例id>     删除实例\n"
                "  /llm rm model <实例id> <模型id>  删除模型\n"
                "注意：删除后立即生效，用 /llm list 查看。"
            )
            return
        sub = tokens[0].lower()
        store = get_store()
        if sub == "source":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llm rm source <站名|源id>")
                return
            target = tokens[1]
            ok = store.remove_source(target)
            if ok:
                yield event.plain_result(
                    f"已删除一级源 {target}（含其下所有分组和模型）。\n"
                    "用 /llm list 查看最新配置。"
                )
            else:
                yield event.plain_result(f"未找到一级源 {target}（用 /llm list 查看站名/源id）。")
            return
        if sub == "instance":
            if len(tokens) < 2:
                yield event.plain_result("用法：/llm rm instance <实例id>")
                return
            ok = store.remove_instance(tokens[1])
            yield event.plain_result(
                f"已删除实例 {tokens[1]}。" if ok else f"未找到实例 {tokens[1]}。"
            )
            return
        if sub == "model":
            if len(tokens) < 3:
                yield event.plain_result("用法：/llm rm model <实例id> <模型id>")
                return
            n = store.remove_models(tokens[1], tokens[2:])
            yield event.plain_result(
                f"实例 {tokens[1]} 移除 {n} 个模型。" if n
                else f"未找到可移除的模型（实例 {tokens[1]}）。"
            )
            return
        yield event.plain_result(f"未知子命令 {sub!r}。支持：source / instance / model")

    # ---------- 运维 ----------

    @llm.command("test")
    async def llm_test(self, event: AstrMessageEvent, arg: str = ""):
        """/llm test <序号|模型id|实例id>  连通性与时延测试。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        target = arg.strip()
        store = get_store()
        item = store.find_in_catalog(target) if target else None
        if item is None:
            yield event.plain_result(
                "用法：/llm test <序号|模型id|实例id>（/llm list 查看序号）"
            )
            return
        from .manager_provider import LLMManagerProvider

        provider = LLMManagerProvider(
            {"model": item["model"], "type": "llm_manager"}, {}
        )
        routed = store.resolve(req_model=item["model"])
        if routed is None:
            yield event.plain_result("该模型当前不可用（源或实例被停用）。")
            return
        backend = provider._backend_for(routed["source"])
        backend.set_model(routed["model_name"])
        start = time.time()
        try:
            await asyncio.wait_for(backend.test(), timeout=60)
            cost = (time.time() - start) * 1000
            yield event.plain_result(
                f"✅ [{item['num']}] {item['model']}（{item['instance_id']}）连通正常，耗时 {cost:.0f} ms"
            )
        except Exception as e:  # noqa: BLE001
            yield event.plain_result(f"❌ [{item['num']}] {item['model']} 测试失败：{e}")

    # ---------- 帮助 ----------

    @llm.command("help")
    async def llm_help(self, event: AstrMessageEvent, arg: str = ""):
        """插件帮助。"""
        deny = self._deny_if_not_admin(event)
        if deny is not None:
            yield deny
            return
        help_text = (
            "LLM 供应商管理插件\n"
            "三级结构：API Base URL → 实例(站名_分组名) → 模型[全局序号]\n\n"
            "查看/切换：\n"
            "  /llm list [关键词] [-t]      三级卡片图片查看；-t 纯文本\n"
            "  /llm use <序号|模型id|实例id> [-c]   全局切换；-c 仅当前会话\n"
            "  /llm use global            清除会话切换，回到全局默认\n"
            "  /llm default <目标>        设置全局默认模型/实例\n"
            "管理（管理员）：\n"
            "  /llm import                 从系统配置(cmd_config.json)导入已有供应商\n"
            "  /llm add <站名> <base_url> <key> [类型]   新增后端源\n"
            "  /llm group add <站名> <分组名> [模型...]   新增分组实例\n"
            "  /llm group rm|enable|disable <实例id>\n"
            "  /llm model add|rm <实例id> <模型id>...    挂载/移除模型\n"
            "运维：\n"
            "  /llm test <目标>           连通性与时延测试\n"
            "配置存放：data/plugin_data/astrbot_plugin_llm_manager/backends.json"
        )
        yield event.plain_result(help_text)
