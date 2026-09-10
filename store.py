"""LLM 供应商统一管理插件 —— 配置仓库与路由核心。

三级结构（与 AstrBot 原生 provider_sources / provider 两级配置对应）：
    一级：API Base URL（即一个 Source：base_url + api_key + 站名）
    二级：提供商源唯一 ID（自动生成为 站名_分组名）
    三级：具体模型 ID

本模块只依赖标准库，便于独立单元测试（不 import astrbot）。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from .stats import Stats

logger = logging.getLogger("astrbot_plugin_llm_manager")

#: 未显式声明能力时，模型默认具备的能力
DEFAULT_MODALITIES = ["text", "tool_use"]

STORE_VERSION = 1


def sanitize_id(raw: str) -> str:
    """把任意文本清洗成可作为命令参数的 ID：保留中英文、数字、下划线、连字符。"""
    s = str(raw).strip()
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s)
    return s


def _normalize_models(models: list[str]) -> list[dict[str, Any]]:
    """把模型名列表规范化为模型条目（补充默认能力声明）。"""
    result: list[dict[str, Any]] = []
    for m in models:
        m = str(m).strip()
        if not m:
            continue
        result.append({"name": m, "modalities": list(DEFAULT_MODALITIES)})
    return result


class Store:
    """JSON 配置仓库：管理 sources -> instances -> models，并提供路由解析。

    线程安全（写操作加锁），读写采用原子替换（临时文件 + rename）。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: dict[str, Any] = self._load()
        # Token 消耗统计（与 backends.json 同目录的 stats.json）
        self.stats = Stats(self.path.parent / "stats.json")

    # ---------- 基础 ----------

    def _default(self) -> dict[str, Any]:
        return {
            "version": STORE_VERSION,
            "default_instance": "",
            "default_model": "",
            "sources": [],
            "overrides": {},  # umo -> instance_id（会话级切换）
        }

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return self._default()

    def save(self) -> None:
        """持久化（原子写）。"""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)

    # ---------- 一级：Source（base_url + key）----------

    def sources(self) -> list[dict[str, Any]]:
        return self.data["sources"]

    def find_source(self, source_id: str) -> dict[str, Any] | None:
        for src in self.sources():
            if src.get("id") == source_id:
                return src
        return None

    def add_source(
        self,
        site: str,
        api_base: str,
        key: str | list[str],
        stype: str = "openai_chat_completion",
        timeout: int = 120,
        proxy: str = "",
        custom_headers: dict | None = None,
    ) -> dict[str, Any]:
        """新增一级源。同站同 base_url 复用；同站不同 base_url 则新增并加后缀。"""
        site = sanitize_id(site)
        api_base = str(api_base).strip().rstrip("/")
        if not site or not api_base:
            raise ValueError("站名与 API Base URL 均不能为空")

        keys = key if isinstance(key, list) else [str(key).strip()] if str(key).strip() else []
        existing = [s for s in self.sources() if s.get("site") == site]
        for src in existing:
            if src.get("api_base") == api_base:
                merged = list(dict.fromkeys((src.get("key") or []) + keys))
                src["key"] = merged
                self.save()
                return src

        base_id = sanitize_id(site)
        source_id = base_id
        n = 2
        while self.find_source(source_id):
            source_id = f"{base_id}-{n}"
            n += 1

        source = {
            "id": source_id,
            "site": site,
            "type": stype,
            "provider_type": "chat_completion",
            "api_base": api_base,
            "key": keys,
            "timeout": int(timeout),
            "proxy": proxy or "",
            "custom_headers": custom_headers or {},
            "enabled": True,
            "instances": [],
        }
        self.sources().append(source)
        self.save()
        return source

    def source_config(self, source: dict[str, Any]) -> dict[str, Any]:
        """把 Source 转为可实例化 AstrBot 内置 Provider 的配置（provider_sources 形态）。"""
        return {
            "id": source.get("id", ""),
            "type": source.get("type", "openai_chat_completion"),
            "provider": source.get("type", "openai_chat_completion"),
            "provider_type": "chat_completion",
            "key": list(source.get("key") or []),
            "api_base": source.get("api_base", ""),
            "timeout": int(source.get("timeout", 120)),
            "proxy": source.get("proxy", "") or "",
            "custom_headers": dict(source.get("custom_headers") or {}),
        }

    # ---------- 二级：Instance（站名_分组名）----------

    def instances(self) -> list[dict[str, Any]]:
        ret: list[dict[str, Any]] = []
        for src in self.sources():
            ret.extend(src.get("instances", []))
        return ret

    def find_instance(self, instance_id: str) -> dict[str, Any] | None:
        for inst in self.instances():
            if inst.get("id") == instance_id:
                return inst
        return None

    def _make_instance_id(self, site: str, group: str) -> str:
        """生成全局唯一的实例 ID：站名_分组名，撞名自动加 -2/-3。"""
        base = f"{sanitize_id(site)}_{sanitize_id(group)}"
        inst_id = base
        n = 2
        while self.find_instance(inst_id):
            inst_id = f"{base}-{n}"
            n += 1
        return inst_id

    def add_instance(
        self,
        site: str,
        group: str,
        models: list[str] | None = None,
    ) -> dict[str, Any]:
        """在指定站名下新增分组实例。模型名可选，随后可用 /llm model add 追加。"""
        site = sanitize_id(site)
        group = str(group).strip()
        if not group:
            raise ValueError("分组名不能为空")
        source = self.find_source(site)
        if source is None:
            raise ValueError(
                f"未找到站名 {site} 的一级源，请先执行 /llm add {site} <base_url> <key>"
            )

        inst_id = self._make_instance_id(site, group)
        instance = {
            "id": inst_id,
            "group": group,
            "enabled": True,
            "models": _normalize_models(models or []),
        }
        source.setdefault("instances", []).append(instance)
        self.save()
        return instance

    def remove_instance(self, instance_id: str) -> bool:
        for src in self.sources():
            for inst in list(src.get("instances", [])):
                if inst.get("id") == instance_id:
                    src["instances"].remove(inst)
                    self._clear_references(instance_id)
                    self.save()
                    return True
        return False

    def remove_source(self, source_id: str) -> bool:
        """删除一级源（包括其下所有实例和模型），并清理引用。"""
        source = self.find_source(source_id)
        if source is None:
            # 也尝试按站名匹配
            for src in self.sources():
                if src.get("site") == source_id:
                    source = src
                    break
        if source is None:
            return False
        # 清理该源下所有实例的引用
        for inst in source.get("instances", []):
            self._clear_references(inst.get("id", ""))
        self.data["sources"].remove(source)
        self.save()
        return True

    def set_instance_enabled(self, instance_id: str, enabled: bool) -> bool:
        inst = self.find_instance(instance_id)
        if inst is None:
            return False
        inst["enabled"] = bool(enabled)
        self.save()
        return True

    def _clear_references(self, instance_id: str) -> None:
        if self.data.get("default_instance") == instance_id:
            self.data["default_instance"] = ""
            self.data["default_model"] = ""
        for umo, val in list(self.data.get("overrides", {}).items()):
            if val == instance_id:
                del self.data["overrides"][umo]

    # ---------- 三级：Model ----------

    def add_models(self, instance_id: str, models: list[str]) -> int:
        inst = self.find_instance(instance_id)
        if inst is None:
            raise ValueError(f"未找到实例 {instance_id}，可先 /llm list 查看现有实例")
        existing = {m["name"] for m in inst.get("models", [])}
        added = 0
        for m in _normalize_models(models):
            if m["name"] not in existing:
                inst.setdefault("models", []).append(m)
                existing.add(m["name"])
                added += 1
        if added:
            self.save()
        return added

    def remove_models(self, instance_id: str, models: list[str]) -> int:
        inst = self.find_instance(instance_id)
        if inst is None:
            return 0
        before = len(inst.get("models", []))
        inst["models"] = [m for m in inst.get("models", []) if m["name"] not in set(models)]
        removed = before - len(inst["models"])
        if removed:
            self.save()
        return removed

    # ---------- 目录与编号 ----------

    def build_catalog(self) -> list[dict[str, Any]]:
        """按 源 -> 实例 -> 模型 顺序生成全局编号目录。

        序号按 api_base 排序后分配（同 api_base 下保持 source 原有顺序），
        与 /llm list 显示顺序一致，避免序号跳号。

        每条：{num, model, modalities, instance_id, source_id, site, api_base,
        source_enabled, instance_enabled, disabled}
        """
        catalog: list[dict[str, Any]] = []
        num = 1
        # 按 api_base 排序（Python sorted 是稳定排序，同 api_base 下保持 source 原有顺序）
        sorted_sources = sorted(self.sources(), key=lambda s: str(s.get("api_base", "")))
        for src in sorted_sources:
            for inst in src.get("instances", []):
                models = inst.get("models", []) or []
                for m in models:
                    disabled = not src.get("enabled", True) or not inst.get("enabled", True)
                    catalog.append(
                        {
                            "num": num,
                            "model": m["name"],
                            "modalities": list(m.get("modalities") or DEFAULT_MODALITIES),
                            "instance_id": inst.get("id", ""),
                            "source_id": src.get("id", ""),
                            "site": src.get("site", ""),
                            "api_base": src.get("api_base", ""),
                            "disabled": disabled,
                        }
                    )
                    num += 1
        return catalog

    def find_in_catalog(self, target: str) -> dict[str, Any] | None:
        """按 序号 / 完整路径(实例ID/模型ID) / 模型 ID / 实例 ID 解析目标（优先级从高到低）。"""
        target = str(target).strip()
        if not target:
            return None
        catalog = self.build_catalog()
        # 1. 纯数字 -> 序号
        if target.isdigit():
            num = int(target)
            for item in catalog:
                if item["num"] == num:
                    return item
            return None
        # 2. 完整路径 实例ID/模型ID（如 deepseek_通用/deepseek-chat）
        if "/" in target:
            inst_id, _, model_name = target.partition("/")
            inst_id, model_name = inst_id.strip(), model_name.strip()
            for item in catalog:
                if item["instance_id"] == inst_id and item["model"] == model_name:
                    return item
            return None
        # 3. 模型 ID
        for item in catalog:
            if item["model"] == target:
                return item
        # 4. 实例 ID（取该实例第一个模型）
        for item in catalog:
            if item["instance_id"] == target:
                return item
        return None

    # ---------- 切换与路由 ----------

    def set_default(self, target: str) -> dict[str, Any] | None:
        """设置全局默认（target 同 find_in_catalog 解析规则）。返回命中的目录项。"""
        item = self.find_in_catalog(target)
        if item is None:
            logger.warning("[LLM Manager] set_default 未找到目标 %r", target)
            return None
        self.data["default_instance"] = item["instance_id"]
        self.data["default_model"] = item["model"]
        self.save()
        logger.info(
            "[LLM Manager] set_default 成功 -> instance=%s model=%s",
            item["instance_id"], item["model"],
        )
        return item

    def set_conversation_override(self, umo: str, target: str) -> dict[str, Any] | None:
        """设置会话级切换。"""
        item = self.find_in_catalog(target)
        if item is None:
            return None
        self.data.setdefault("overrides", {})[umo] = item["instance_id"]
        self.save()
        return item

    def clear_conversation_override(self, umo: str) -> bool:
        overrides = self.data.setdefault("overrides", {})
        if umo in overrides:
            del overrides[umo]
            self.save()
            return True
        return False

    def get_conversation_override(self, umo: str) -> str | None:
        return self.data.setdefault("overrides", {}).get(umo)

    def resolve(
        self,
        req_model: str | None = None,
        umo: str | None = None,
    ) -> dict[str, Any] | None:
        """路由解析：返回 {source, instance, model_name, model_entry} 或 None。

        优先级：
          1. 请求显式指定的模型（命中目录时）
          2. 会话级 override（umo）
          3. 全局默认（default_instance + default_model）
          4. 第一个启用的实例
        """
        catalog = self.build_catalog()

        def _usable(item: dict[str, Any]) -> bool:
            return not item["disabled"]

        # 1. 请求显式模型
        if req_model:
            for item in catalog:
                if item["model"] == req_model and _usable(item):
                    return self._to_routed(item)

        # 2. 会话级 override
        if umo:
            override_id = self.get_conversation_override(umo)
            if override_id:
                for item in catalog:
                    if item["instance_id"] == override_id and _usable(item):
                        return self._to_routed(item)

        # 3. 全局默认
        default_instance = self.data.get("default_instance", "")
        default_model = self.data.get("default_model", "")
        if default_instance:
            candidates = [i for i in catalog if i["instance_id"] == default_instance]
            if default_model:
                for item in candidates:
                    if item["model"] == default_model and _usable(item):
                        return self._to_routed(item)
            for item in candidates:
                if _usable(item):
                    return self._to_routed(item)

        # 4. 兜底：第一个启用的模型
        for item in catalog:
            if _usable(item):
                return self._to_routed(item)
        return None

    def _to_routed(self, item: dict[str, Any]) -> dict[str, Any]:
        source = self.find_source(item["source_id"])
        inst = self.find_instance(item["instance_id"])
        model_entry = {"name": item["model"], "modalities": item["modalities"]}
        return {
            "source": source,
            "instance": inst,
            "model_name": item["model"],
            "model_entry": model_entry,
        }

    # ---------- 从 AstrBot 系统配置导入 ----------

    def import_from_astrbot_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """从 AstrBot cmd_config.json 导入 provider_sources 和 provider。

        映射关系：
          provider_sources → Source（site=provider 字段, api_base, key, type...）
          provider 按 provider_source_id 分组 → Instance（id=源ID, group="默认"）
          provider.model → Model（带 modalities）

        跳过没有 provider_source_id 的条目（如 dashscope agent_runner）。
        幂等：已存在的 source/instance/model 不重复创建。
        """
        sources_cfg = config.get("provider_sources", []) or []
        providers_cfg = config.get("provider", []) or []

        providers_by_source: dict[str, list[dict[str, Any]]] = {}
        skipped: list[str] = []
        for p in providers_cfg:
            psid = p.get("provider_source_id")
            if not psid:
                skipped.append(str(p.get("id", "?")))
                continue
            providers_by_source.setdefault(psid, []).append(p)

        imported_sources = 0
        imported_instances = 0
        imported_models = 0

        for src_cfg in sources_cfg:
            src_id = str(src_cfg.get("id", "")).strip()
            api_base = str(src_cfg.get("api_base") or "").rstrip("/")
            if not src_id or not api_base:
                continue

            # 查找或创建 Source（按 provider_source 的 id 匹配，每个 provider_source 都是独立的 source，
            # 保留自己的 key——即使 api_base 相同，不同 key 对应中转后台不同的分组权限）
            source = self.find_source(src_id)

            if source is None:
                source = {
                    "id": src_id,
                    "site": str(src_cfg.get("provider", src_id)),
                    "type": str(src_cfg.get("type", "openai_chat_completion")),
                    "provider_type": str(src_cfg.get("provider_type", "chat_completion")),
                    "api_base": api_base,
                    "key": list(src_cfg.get("key", []) or []),
                    "timeout": int(src_cfg.get("timeout", 120)),
                    "proxy": str(src_cfg.get("proxy", "")),
                    "custom_headers": dict(src_cfg.get("custom_headers", {}) or {}),
                    "enabled": bool(src_cfg.get("enable", True)),
                    "instances": [],
                }
                self.data["sources"].append(source)
                imported_sources += 1
            else:
                # 已存在的 source，同步更新 key/api_base/type/启停（防止用户在 WebUI 改了配置）
                source["key"] = list(src_cfg.get("key", []) or [])
                source["api_base"] = api_base
                source["type"] = str(src_cfg.get("type", "openai_chat_completion"))
                source["provider_type"] = str(src_cfg.get("provider_type", "chat_completion"))
                source["timeout"] = int(src_cfg.get("timeout", 120))
                source["proxy"] = str(src_cfg.get("proxy", ""))
                source["custom_headers"] = dict(src_cfg.get("custom_headers", {}) or {})
                source["enabled"] = bool(src_cfg.get("enable", True))

            # 查找或创建 Instance（id=源ID，即提供商源唯一 ID）
            instance = None
            for inst in source.get("instances", []):
                if inst.get("id") == src_id:
                    instance = inst
                    break

            if instance is None:
                instance = {
                    "id": src_id,
                    "group": "默认",
                    "enabled": True,
                    "models": [],
                }
                source.setdefault("instances", []).append(instance)
                imported_instances += 1

            # 收集该源下所有 provider 的模型
            existing_models = {m["name"] for m in instance.get("models", [])}
            for p in providers_by_source.get(src_id, []):
                model_name = str(p.get("model", "")).strip()
                if not model_name or model_name in existing_models:
                    continue
                modalities = p.get("modalities") or DEFAULT_MODALITIES
                instance.setdefault("models", []).append({
                    "name": model_name,
                    "modalities": list(modalities),
                })
                existing_models.add(model_name)
                imported_models += 1

            # 实例启停：关联的 provider 至少一个启用则实例启用
            linked = providers_by_source.get(src_id, [])
            if linked:
                instance["enabled"] = any(p.get("enable", True) for p in linked)

        # ---------- 自动保留原默认模型 ----------
        # 读取 cmd_config.json 里原来的 default_provider_id，解析出源ID/模型名，
        # 在插件目录中匹配并设为全局默认，避免安装插件后默认模型被兜底逻辑改掉。
        original_default = ""
        auto_default_set = False
        try:
            provider_settings = config.get("provider_settings", {}) or {}
            original_default = str(
                provider_settings.get("default_provider_id", "") or ""
            )
            # 兜底：default_provider_id 为空时尝试从 provider_pool 取第一个
            if not original_default:
                pool = provider_settings.get("provider_pool", []) or []
                if pool:
                    first = pool[0]
                    if isinstance(first, str):
                        original_default = first
                    elif isinstance(first, dict):
                        original_default = str(
                            first.get("id", "") or first.get("provider_id", "")
                        )
        except Exception:  # noqa: BLE001
            original_default = ""

        if (
            original_default
            and "/" in original_default
            and not self.data.get("default_instance", "")  # 不覆盖用户已设的默认
        ):
            orig_src_id, _, orig_model = original_default.partition("/")
            orig_src_id, orig_model = orig_src_id.strip(), orig_model.strip()
            # 排除指向 llm_manager 自己的情况（用户已把默认改成插件自身）
            is_self = any(
                str(s.get("id", "")) == orig_src_id
                and str(s.get("type", "")) == "llm_manager"
                for s in sources_cfg
            )
            if not is_self:
                for item in self.build_catalog():
                    if (
                        item["instance_id"] == orig_src_id
                        and item["model"] == orig_model
                        and not item["disabled"]
                    ):
                        self.data["default_instance"] = item["instance_id"]
                        self.data["default_model"] = item["model"]
                        auto_default_set = True
                        break

        self.save()
        return {
            "imported_sources": imported_sources,
            "imported_instances": imported_instances,
            "imported_models": imported_models,
            "skipped": skipped,
            "total_sources": len(self.data["sources"]),
            "total_models": len(self.build_catalog()),
            "original_default": original_default,
            "auto_default_set": auto_default_set,
        }

    # ---------- 手动同步（解决手动修改 cmd_config.json 后不同步的问题）----------

    def resync_from_astrbot_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """从 AstrBot 配置完全重建 sources（先清空再导入）。

        用于用户手动在 WebUI 删除/修改供应商后，让插件配置与系统配置同步。
        保留 default_instance/default_model（如果目标仍存在），清理无效 overrides。
        """
        # 1. 保存当前设置
        old_default_instance = self.data.get("default_instance", "")
        old_default_model = self.data.get("default_model", "")
        old_overrides = dict(self.data.get("overrides", {}))
        old_source_count = len(self.data["sources"])
        old_model_count = len(self.build_catalog())

        # 2. 清空 sources（保留 default/overrides 字段，后面恢复）
        self.data["sources"] = []

        # 3. 重新导入（import 是幂等的，此时 sources 为空，会全部新建）
        result = self.import_from_astrbot_config(config)

        # 4. 恢复 default（优先精确匹配 实例+模型，其次匹配实例，找不到则清空）
        default_preserved = False
        if old_default_instance:
            catalog = self.build_catalog()
            # 精确匹配
            for item in catalog:
                if item["instance_id"] == old_default_instance and item["model"] == old_default_model:
                    self.data["default_instance"] = old_default_instance
                    self.data["default_model"] = old_default_model
                    default_preserved = True
                    break
            # 只匹配实例（用该实例第一个模型）
            if not default_preserved:
                for item in catalog:
                    if item["instance_id"] == old_default_instance:
                        self.data["default_instance"] = old_default_instance
                        self.data["default_model"] = item["model"]
                        default_preserved = True
                        break
            if not default_preserved:
                self.data["default_instance"] = ""
                self.data["default_model"] = ""

        # 5. 清理无效 overrides（实例已被删除的）
        valid_instances = {inst["id"] for inst in self.instances()}
        cleaned_overrides = {
            umo: iid for umo, iid in old_overrides.items()
            if iid in valid_instances
        }
        self.data["overrides"] = cleaned_overrides
        overrides_cleaned = len(old_overrides) - len(cleaned_overrides)

        # 6. 保存
        self.save()

        return {
            "old_sources": old_source_count,
            "old_models": old_model_count,
            "new_sources": result["total_sources"],
            "new_models": result["total_models"],
            "default_preserved": default_preserved,
            "overrides_cleaned": overrides_cleaned,
            "skipped": result["skipped"],
        }
