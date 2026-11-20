"""验证虚拟 Provider 的缓存失效与流式 token 记录。"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

from astrbot_plugin_llm_manager import manager_provider  # noqa: E402
from astrbot_plugin_llm_manager.store import Store  # noqa: E402


class FakeBackend:
    def __init__(self, cfg, settings):
        self.provider_config = dict(cfg)
        self.provider_settings = settings
        self.model = str(cfg.get("model") or "")
        self.stream_calls = []

    def set_model(self, model):
        self.model = str(model)

    async def text_chat(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        )

    async def text_chat_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        yield "a"
        yield "b"
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4, total_tokens=11)
        )

    async def test(self):
        return True


class ManagerProviderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.store.add_source("A", "https://a.example/v1", "old-key", stype="fake")
        self.store.add_instance("A", "通用", ["m1", "m2"])
        self.store.set_default("m2")
        manager_provider.set_store(self.store)
        manager_provider.provider_cls_map["fake"] = SimpleNamespace(cls_type=FakeBackend)
        manager_provider.set_fallback_timeout(None)
        self.addCleanup(manager_provider.set_fallback_timeout, None)
        self.provider = manager_provider.LLMManagerProvider(
            {"model": "main", "type": "llm_manager"}, {}
        )

    def test_backend_cache_invalidated_when_config_changes(self):
        source = self.store.find_source("A")
        first = self.provider._backend_for(source)
        self.assertIs(first, self.provider._backend_for(source))
        source["key"] = ["new-key"]
        second = self.provider._backend_for(source)
        self.assertIsNot(first, second)
        self.assertEqual(second.provider_config["key"], ["new-key"])
        self.assertEqual(len(self.provider._backend_cache), 1)

    def test_backend_timeout_override_is_cache_keyed(self):
        source = self.store.find_source("A")
        default = self.provider._backend_for(source)
        self.assertEqual(default.provider_config["timeout"], 120)

        fast = self.provider._backend_for(source, 90)
        self.assertIsNot(default, fast)
        self.assertEqual(fast.provider_config["timeout"], 90)
        self.assertIs(fast, self.provider._backend_for(source, 90))
        self.assertEqual(len(self.provider._backend_cache), 1)

    def test_backend_cache_separates_models(self):
        source = self.store.find_source("A")
        first = self.provider._backend_for(source, None, "m1")
        second = self.provider._backend_for(source, None, "m2")
        self.assertIsNot(first, second)
        self.assertIs(first, self.provider._backend_for(source, None, "m1"))
        self.assertEqual(len(self.provider._backend_cache), 2)

    def test_set_fallback_timeout_normalizes(self):
        manager_provider.set_fallback_timeout("90")
        self.assertEqual(manager_provider._fallback_timeout, 90)
        for value in (0, None, "bad", -5):
            manager_provider.set_fallback_timeout(value)
            self.assertIsNone(manager_provider._fallback_timeout, value)

    def test_get_store_refreshes_external_writes(self):
        """路由侧 Store 应看到另一个实例（指令侧）落盘的会话切换。"""
        other = Store(self.store.path)
        other.set_conversation_override("sess-ext", "m1")
        self.assertIsNone(self.store.get_conversation_override("sess-ext"))

        self.assertIs(manager_provider.get_store(), self.store)
        self.assertEqual(
            self.store.get_conversation_override("sess-ext")["model"], "m1"
        )

        # 反向：路由侧清除后，另一个实例刷新也能立即看到
        self.store.clear_conversation_override("sess-ext")
        self.assertTrue(other.refresh_if_changed())
        self.assertIsNone(other.get_conversation_override("sess-ext"))

    def test_stream_records_usage_once(self):
        async def collect():
            return [chunk async for chunk in self.provider.text_chat_stream(prompt="hi")]

        chunks = asyncio.run(collect())
        self.assertEqual(chunks[:2], ["a", "b"])
        self.assertEqual(self.store.stats.record_count(), 1)
        self.assertEqual(self.store.stats.total_tokens(1), 11)
        self.assertEqual(self.store.stats.total_calls(1), 1)

    def test_non_stream_records_usage(self):
        asyncio.run(self.provider.text_chat(prompt="hi"))
        self.assertEqual(self.store.stats.record_count(), 1)
        self.assertEqual(self.store.stats.total_tokens(1), 5)

    def test_verbose_routing_log_toggle(self):
        records = []

        class Recorder:
            def info(self, message, *args):
                records.append(("info", message))

            def debug(self, message, *args):
                records.append(("debug", message))

        original = manager_provider.logger
        manager_provider.logger = Recorder()
        self.addCleanup(setattr, manager_provider, "logger", original)
        self.addCleanup(manager_provider.set_verbose_logging, True)
        try:
            manager_provider.set_verbose_logging(True)
            manager_provider._routing_log("enabled")
            manager_provider.set_verbose_logging(False)
            manager_provider._routing_log("disabled")
        finally:
            manager_provider.set_verbose_logging(True)

        self.assertEqual(
            [(level, message) for level, message in records],
            [("info", "enabled"), ("debug", "disabled")],
        )


if __name__ == "__main__":
    unittest.main()
