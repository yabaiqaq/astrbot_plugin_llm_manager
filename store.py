"""LLM 供应商统一管理插件 —— 配置仓库与路由核心。"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("astrbot_plugin_llm_manager")

DEFAULT_MODALITIES = ["text", "tool_use"]
STORE_VERSION = 1


def sanitize_id(raw):
    s = str(raw).strip()
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", s)
    return s


def _normalize_models(models):
    result = []
    for m in models:
        m = str(m).strip()
        if not m:
            continue
        result.append({"name": m, "modalities": list(DEFAULT_MODALITIES)})
    return result


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data = self._load()

    def _default(self):
        return {
            "version": STORE_VERSION,
            "default_instance": "",
            "default_model": "",
            "sources": [],
            "overrides": {},
        }

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return self._default()

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def sources(self):
        return self.data["sources"]

    def find_source(self, source_id):
        for src in self.sources():
            if src.get("id") == source_id:
                return src
        return None

    def add_source(self, site, api_base, key, stype="openai_chat_completion", timeout=120, proxy="", custom_headers=None):
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
            "id": source_id, "site": site, "type": stype, "provider_type": "chat_completion",
            "api_base": api_base, "key": keys, "timeout": int(timeout),
            "proxy": proxy or "", "custom_headers": custom_headers or {}, "enabled": True, "instances": [],
        }
        self.sources().append(source)
        self.save()
        return source

    def source_config(self, source):
        return {
            "id": source.get("id", ""), "type": source.get("type", "openai_chat_completion"),
            "provider": source.get("type", "openai_chat_completion"), "provider_type": "chat_completion",
            "key": list(source.get("key") or []), "api_base": source.get("api_base", ""),
            "timeout": int(source.get("timeout", 120)), "proxy": source.get("proxy", "") or "",
            "custom_headers": dict(source.get("custom_headers") or {}),
        }

    def instances(self):
        ret = []
        for src in self.sources():
            ret.extend(src.get("instances", []))
        return ret

    def find_instance(self, instance_id):
        for inst in self.instances():
            if inst.get("id") == instance_id:
                return inst
        return None

    def _make_instance_id(self, site, group):
        base = f"{sanitize_id(site)}_{sanitize_id(group)}"
        inst_id = base
        n = 2
        while self.find_instance(inst_id):
            inst_id = f"{base}-{n}"
            n += 1
        return inst_id

    def add_instance(self, site, group, models=None):
        site = sanitize_id(site)
        group = str(group).strip()
        if not group:
            raise ValueError("分组名不能为空")
        source = self.find_source(site)
        if source is None:
            raise ValueError(f"未找到站名 {site} 的一级源，请先执行 /llm add {site} <base_url> <key>")
        inst_id = self._make_instance_id(site, group)
        instance = {"id": inst_id, "group": group, "enabled": True, "models": _normalize_models(models or [])}
        source.setdefault("instances", []).append(instance)
        self.save()
        return instance

    def remove_instance(self, instance_id):
        for src in self.sources():
            for inst in list(src.get("instances", [])):
                if inst.get("id") == instance_id:
                    src["instances"].remove(inst)
                    self._clear_references(instance_id)
                    self.save()
                    return True
        return False

    def remove_source(self, source_id):
        source = self.find_source(source_id)
        if source is None:
            for src in self.sources():
                if src.get("site") == source_id:
                    source = src
                    break
        if source is None:
            return False
        for inst in source.get("instances", []):
            self._clear_references(inst.get("id", ""))
        self.data["sources"].remove(source)
        self.save()
        return True

    def set_instance_enabled(self, instance_id, enabled):
        inst = self.find_instance(instance_id)
        if inst is None:
            return False
        inst["enabled"] = bool(enabled)
        self.save()
        return True

    def _clear_references(self, instance_id):
        if self.data.get("default_instance") == instance_id:
            self.data["default_instance"] = ""
            self.data["default_model"] = ""
        for umo, val in list(self.data.get("overrides", {}).items()):
            if val == instance_id:
                del self.data["overrides"][umo]

    def add_models(self, instance_id, models):
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

    def remove_models(self, instance_id, models):
        inst = self.find_instance(instance_id)
        if inst is None:
            return 0
        before = len(inst.get("models", []))
        inst["models"] = [m for m in inst.get("models", []) if m["name"] not in set(models)]
        removed = before - len(inst["models"])
        if removed:
            self.save()
        return removed

    def build_catalog(self):
        catalog = []
        num = 1
        sorted_sources = sorted(self.sources(), key=lambda s: str(s.get("api_base", "")))
        for src in sorted_sources:
            for inst in src.get("instances", []):
                models = inst.get("models", []) or []
                for m in models:
                    disabled = not src.get("enabled", True) or not inst.get("enabled", True)
                    catalog.append({
                        "num": num, "model": m["name"], "modalities": list(m.get("modalities") or DEFAULT_MODALITIES),
                        "instance_id": inst.get("id", ""), "source_id": src.get("id", ""),
                        "site": src.get("site", ""), "api_base": src.get("api_base", ""), "disabled": disabled,
                    })
                    num += 1
        return catalog

    def find_in_catalog(self, target):
        target = str(target).strip()
        if not target:
            return None
        catalog = self.build_catalog()
        if target.isdigit():
            num = int(target)
            for item in catalog:
                if item["num"] == num:
                    return item
            return None
        if "/" in target:
            inst_id, _, model_name = target.partition("/")
            inst_id, model_name = inst_id.strip(), model_name.strip()
            for item in catalog:
                if item["instance_id"] == inst_id and item["model"] == model_name:
                    return item
            return None
        for item in catalog:
            if item["model"] == target:
                return item
        for item in catalog:
            if item["instance_id"] == target:
                return item
        return None

    def set_default(self, target):
        item = self.find_in_catalog(target)
        if item is None:
            logger.warning("[LLM Manager] set_default 未找到目标 %r", target)
            return None
        self.data["default_instance"] = item["instance_id"]
        self.data["default_model"] = item["model"]
        self.save()
        logger.info("[LLM Manager] set_default 成功 -> instance=%s model=%s", item["instance_id"], item["model"])
        return item

    def set_conversation_override(self, umo, target):
        item = self.find_in_catalog(target)
        if item is None:
            return None
        self.data.setdefault("overrides", {})[umo] = item["instance_id"]
        self.save()
        return item

    def clear_conversation_override(self, umo):
        overrides = self.data.setdefault("overrides", {})
        if umo in overrides:
            del overrides[umo]
            self.save()
            return True
        return False

    def get_conversation_override(self, umo):
        return self.data.setdefault("overrides", {}).get(umo)

    def resolve(self, req_model=None, umo=None):
        catalog = self.build_catalog()
        def _usable(item):
            return not item["disabled"]
        if req_model:
            for item in catalog:
                if item["model"] == req_model and _usable(item):
                    return self._to_routed(item)
        if umo:
            override_id = self.get_conversation_override(umo)
            if override_id:
                for item in catalog:
                    if item["instance_id"] == override_id and _usable(item):
                        return self._to_routed(item)
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
        for item in catalog:
            if _usable(item):
                return self._to_routed(item)
        return None

    def _to_routed(self, item):
        source = self.find_source(item["source_id"])
        inst = self.find_instance(item["instance_id"])
        model_entry = {"name": item["model"], "modalities": item["modalities"]}
        return {"source": source, "instance": inst, "model_name": item["model"], "model_entry": model_entry}

    def import_from_astrbot_config(self, config):
        sources_cfg = config.get("provider_sources", []) or []
        providers_cfg = config.get("provider", []) or []
        providers_by_source = {}
        skipped = []
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
            source = self.find_source(src_id)
            if source is None:
                source = {
                    "id": src_id, "site": str(src_cfg.get("provider", src_id)),
                    "type": str(src_cfg.get("type", "openai_chat_completion")),
                    "provider_type": str(src_cfg.get("provider_type", "chat_completion")),
                    "api_base": api_base, "key": list(src_cfg.get("key", []) or []),
                    "timeout": int(src_cfg.get("timeout", 120)), "proxy": str(src_cfg.get("proxy", "")),
                    "custom_headers": dict(src_cfg.get("custom_headers", {}) or {}),
                    "enabled": bool(src_cfg.get("enable", True)), "instances": [],
                }
                self.data["sources"].append(source)
                imported_sources += 1
            else:
                source["key"] = list(src_cfg.get("key", []) or [])
                source["api_base"] = api_base
                source["type"] = str(src_cfg.get("type", "openai_chat_completion"))
                source["provider_type"] = str(src_cfg.get("provider_type", "chat_completion"))
                source["timeout"] = int(src_cfg.get("timeout", 120))
                source["proxy"] = str(src_cfg.get("proxy", ""))
                source["custom_headers"] = dict(src_cfg.get("custom_headers", {}) or {})
                source["enabled"] = bool(src_cfg.get("enable", True))
            instance = None
            for inst in source.get("instances", []):
                if inst.get("id") == src_id:
                    instance = inst
                    break
            if instance is None:
                instance = {"id": src_id, "group": "默认", "enabled": True, "models": []}
                source.setdefault("instances", []).append(instance)
                imported_instances += 1
            existing_models = {m["name"] for m in instance.get("models", [])}
            for p in providers_by_source.get(src_id, []):
                model_name = str(p.get("model", "")).strip()
                if not model_name or model_name in existing_models:
                    continue
                modalities = p.get("modalities") or DEFAULT_MODALITIES
                instance.setdefault("models", []).append({"name": model_name, "modalities": list(modalities)})
                existing_models.add(model_name)
                imported_models += 1
            linked = providers_by_source.get(src_id, [])
            if linked:
                instance["enabled"] = any(p.get("enable", True) for p in linked)
        original_default = ""
        auto_default_set = False
        try:
            provider_settings = config.get("provider_settings", {}) or {}
            original_default = str(provider_settings.get("default_provider_id", "") or "")
            if not original_default:
                pool = provider_settings.get("provider_pool", []) or []
                if pool:
                    first = pool[0]
                    if isinstance(first, str):
                        original_default = first
                    elif isinstance(first, dict):
                        original_default = str(first.get("id", "") or first.get("provider_id", ""))
        except Exception:
            original_default = ""
        if original_default and "/" in original_default and not self.data.get("default_instance", ""):
            orig_src_id, _, orig_model = original_default.partition("/")
            orig_src_id, orig_model = orig_src_id.strip(), orig_model.strip()
            is_self = any(
                str(s.get("id", "")) == orig_src_id and str(s.get("type", "")) == "llm_manager"
                for s in sources_cfg
            )
            if not is_self:
                for item in self.build_catalog():
                    if item["instance_id"] == orig_src_id and item["model"] == orig_model and not item["disabled"]:
                        self.data["default_instance"] = item["instance_id"]
                        self.data["default_model"] = item["model"]
                        auto_default_set = True
                        break
        self.save()
        return {
            "imported_sources": imported_sources, "imported_instances": imported_instances,
            "imported_models": imported_models, "skipped": skipped,
            "total_sources": len(self.data["sources"]), "total_models": len(self.build_catalog()),
            "original_default": original_default, "auto_default_set": auto_default_set,
        }

    def resync_from_astrbot_config(self, config):
        old_default_instance = self.data.get("default_instance", "")
        old_default_model = self.data.get("default_model", "")
        old_overrides = dict(self.data.get("overrides", {}))
        old_source_count = len(self.data["sources"])
        old_model_count = len(self.build_catalog())
        self.data["sources"] = []
        result = self.import_from_astrbot_config(config)
        default_preserved = False
        if old_default_instance:
            catalog = self.build_catalog()
            for item in catalog:
                if item["instance_id"] == old_default_instance and item["model"] == old_default_model:
                    self.data["default_instance"] = old_default_instance
                    self.data["default_model"] = old_default_model
                    default_preserved = True
                    break
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
        valid_instances = {inst["id"] for inst in self.instances()}
        cleaned_overrides = {umo: iid for umo, iid in old_overrides.items() if iid in valid_instances}
        self.data["overrides"] = cleaned_overrides
        overrides_cleaned = len(old_overrides) - len(cleaned_overrides)
        self.save()
        return {
            "old_sources": old_source_count, "old_models": old_model_count,
            "new_sources": result["total_sources"], "new_models": result["total_models"],
            "default_preserved": default_preserved, "overrides_cleaned": overrides_cleaned,
            "skipped": result["skipped"],
        }
