"""虚拟 Provider 故障转移、错误分类、冷却与 sticky 测试。"""
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
from astrbot_plugin_llm_manager.manager_provider import _classify_error  # noqa: E402
from astrbot_plugin_llm_manager.store import Store  # noqa: E402


class HTTPError(Exception):
    def __init__(self, status: int, message: str = "") -> None:
        self.status_code = status
        super().__init__(message or f"http {status}")


class FakeBackend:
    text_failures: dict[str, object] = {}
    stream_failures: dict[str, object] = {}
    calls: list[tuple[str, str, str]] = []
    kwargs_log: list[dict] = []
    instances: list["FakeBackend"] = []

    def __init__(self, cfg, settings) -> None:
        self.provider_config = dict(cfg)
        self.provider_settings = settings
        self.model = str(cfg.get("model") or "")
        FakeBackend.instances.append(self)

    def set_model(self, model: str) -> None:
        self.model = str(model)

    async def text_chat(self, **kwargs):
        model = str(kwargs.get("model") or self.model)
        FakeBackend.calls.append(
            ("text", str(self.provider_config.get("id", "")), model)
        )
        FakeBackend.kwargs_log.append(dict(kwargs))
        failure = FakeBackend.text_failures.get(model)
        if failure is not None:
            raise failure() if callable(failure) else failure
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        )

    async def text_chat_stream(self, **kwargs):
        model = str(kwargs.get("model") or self.model)
        FakeBackend.calls.append(
            ("stream", str(self.provider_config.get("id", "")), model)
        )
        FakeBackend.kwargs_log.append(dict(kwargs))
        failure = FakeBackend.stream_failures.get(model)
        if failure == "before":
            raise HTTPError(503)
        yield "ok"
        if failure == "after":
            raise HTTPError(503)
        yield SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=4, total_tokens=11)
        )

    async def test(self) -> bool:
        return True


class FailoverProviderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.store.add_source("A", "https://a.example/v1", "ka", stype="fake")
        self.store.add_instance("A", "通用", ["m1", "m2"])
        self.store.add_source("B", "https://b.example/v1", "kb", stype="fake")
        self.store.add_instance("B", "通用", ["n1"])
        self.store.set_default("m1")
        manager_provider.set_store(self.store)
        manager_provider.provider_cls_map["fake"] = SimpleNamespace(
            cls_type=FakeBackend
        )
        manager_provider.set_fallback_timeout(None)
        self.addCleanup(manager_provider.set_fallback_timeout, None)
        FakeBackend.text_failures = {}
        FakeBackend.stream_failures = {}
        FakeBackend.calls = []
        FakeBackend.kwargs_log = []
        FakeBackend.instances = []
        self.provider = manager_provider.LLMManagerProvider(
            {"model": "main", "type": "llm_manager"}, {}
        )

    def enable_pool(self, targets=("2", "B_通用/n1"), mode="transient", max_attempts=3):
        self.store.add_fallback_targets(list(targets))
        self.store.set_fallback_enabled(True)
        self.store.set_fallback_mode(mode)
        self.store.set_fallback_max(max_attempts)

    def set_modalities(self, model: str, modalities: list[str]) -> None:
        for inst in self.store.instances():
            for entry in inst.get("models", []):
                if entry["name"] == model:
                    entry["modalities"] = list(modalities)
        self.store.save()

    def test_non_stream_falls_back_before_output(self):
        self.enable_pool()
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        response = asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertIsNotNone(response)
        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1", "m2"]
        )
        self.assertEqual(self.store.stats.record_count(), 1)
        self.assertEqual(self.store.stats.fallback_calls(1), 1)
        self.assertTrue(self.store.is_cooling("A_通用", "m1", "A"))
        self.assertFalse(self.store.is_cooling("A_通用", "m2", "A"))

        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi"))
        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1", "m2"],
            "主模型每轮都先探测一次，恢复后不会继续被冷却跳过",
        )

    def test_cooling_fallback_is_skipped(self):
        self.enable_pool()
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        FakeBackend.text_failures["m2"] = lambda: HTTPError(503)
        FakeBackend.text_failures["n1"] = lambda: HTTPError(503)

        with self.assertRaises(HTTPError):
            asyncio.run(self.provider.text_chat(prompt="hi"))
        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1", "m2", "n1"]
        )

        FakeBackend.text_failures = {}
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi"))
        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1"],
            "冷却中的备选应被跳过，主模型成功后不再尝试备选",
        )

    def test_non_retryable_error_does_not_fall_back(self):
        self.enable_pool()
        FakeBackend.text_failures["m1"] = lambda: HTTPError(400)

        with self.assertRaises(HTTPError):
            asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])
        self.assertFalse(self.store.is_cooling("A_通用", "m1", "A"))
        self.assertEqual(self.store.stats.record_count(), 0)

    def test_auth_error_cools_source_and_switches_source(self):
        self.enable_pool()
        FakeBackend.text_failures["m1"] = lambda: HTTPError(401)

        asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1", "n1"]
        )
        self.assertTrue(self.store.is_cooling("A_通用", "m2", "A"))
        self.assertFalse(self.store.is_cooling("B_通用", "n1", "B"))
        self.assertEqual(self.store.stats.fallback_calls(1), 1)

    def test_capability_mismatch_skips_fallback(self):
        self.set_modalities("m1", ["text", "image"])
        self.set_modalities("m2", ["text"])
        self.set_modalities("n1", ["text"])
        self.enable_pool()
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        with self.assertRaises(HTTPError):
            asyncio.run(
                self.provider.text_chat(prompt="hi", image_urls=["data:image/png;base64,x"])
            )

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])

    def test_stream_falls_back_before_first_chunk(self):
        self.enable_pool()
        FakeBackend.stream_failures["m1"] = "before"

        async def collect():
            return [chunk async for chunk in self.provider.text_chat_stream(prompt="hi")]

        chunks = asyncio.run(collect())
        self.assertEqual(
            [c for c in chunks if isinstance(c, str)], ["ok"]
        )
        self.assertEqual(
            [call[2] for call in FakeBackend.calls], ["m1", "m2"]
        )
        self.assertEqual(self.store.stats.record_count(), 1)
        self.assertEqual(self.store.stats.total_tokens(1), 11)
        self.assertEqual(self.store.stats.fallback_calls(1), 1)

    def test_stream_does_not_fall_back_after_first_chunk(self):
        self.enable_pool()
        FakeBackend.stream_failures["m1"] = "after"

        async def collect():
            return [chunk async for chunk in self.provider.text_chat_stream(prompt="hi")]

        with self.assertRaises(HTTPError):
            asyncio.run(collect())
        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])
        self.assertEqual(self.store.stats.record_count(), 0)

    def test_sticky_degrades_globally_for_all_sessions(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "m2"])
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "A_通用", "model": "m2"},
        )
        self.assertIsNone(self.store.get_conversation_override("sess-1"))
        self.assertEqual(self.store.data["default_model"], "m1")

        # 其他会话不必再探测主模型，直接使用全局降级目标
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-2"))
        self.assertEqual([call[2] for call in FakeBackend.calls], ["m2"])

    def test_degraded_target_without_image_capability_falls_through_to_pool(self):
        self.store.add_models("A_通用", ["m3"])
        self.set_modalities("m1", ["text", "image", "tool_use"])
        self.set_modalities("m2", ["text", "tool_use"])
        self.set_modalities("m3", ["text", "image", "tool_use"])
        self.enable_pool(targets=("2", "3"), mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        # 先由文本请求建立全局降级：m1 -> m2
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual(self.store.degraded_model()["model"], "m2")

        # 图片请求：m2 不支持图片，本次不能粘滞到它，应沿能力匹配的池继续
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(
            prompt="hi", session_id="sess-2", image_urls=["https://img.example/a.png"]
        ))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "m3"])

    def test_sticky_cascades_to_next_pool_entry(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        # 降级目标也失败：继续沿备选池下移，并更新全局降级
        FakeBackend.text_failures["m2"] = lambda: HTTPError(503)
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-2"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m2", "n1"])
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "B_通用", "model": "n1"},
        )

    def test_degraded_target_cooling_switches_to_next_pool_entry(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual(self.store.degraded_model()["model"], "m2")

        # 降级目标 m2 也失败进入冷却：下一轮不应再等它超时，直接换下一个
        self.store.mark_cooldown("A_通用", "m2", "A", 120, "model")
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-2"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["n1"])
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "B_通用", "model": "n1"},
        )

    def test_sticky_reverts_after_primary_recovers(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        # 主模型冷却结束且已恢复：下一轮应自动回到主模型并清除全局降级
        self.store.reset_cooldowns()
        FakeBackend.text_failures = {}
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])
        self.assertIsNone(self.store.degraded_model())

    def test_move_pool_clears_degraded_and_applies_new_order(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "A_通用", "model": "m2"},
        )

        # 把 m2 挪到池尾：全局降级应被清除，下一次请求按新顺序 n1 优先
        self.assertTrue(self.store.move_fallback(1, 2))
        self.assertIsNone(self.store.degraded_model())

        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "n1"])

    def test_new_pool_entry_moved_before_degraded_takes_over(self):
        """复现：降级到旧备选后，新增模型并调到它前面，应立即改用新模型。"""
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual(self.store.degraded_model()["model"], "m2")

        self.store.add_models("A_通用", ["m3"])  # 目录里新增一个模型
        self.store.add_fallback_targets(["3"])  # 按序号追加 m3
        self.assertTrue(self.store.move_fallback(3, 1))  # 提到 m2 之前        self.assertIsNone(self.store.degraded_model())

        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "m3"])
        self.assertEqual(self.store.degraded_model()["model"], "m3")

    def test_move_pool_clears_stale_cooldown_so_new_order_applies(self):
        self.enable_pool()
        self.store.mark_cooldown("B_通用", "n1", "B", 120, "model")
        self.assertTrue(self.store.move_fallback(1, 2))
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi"))
        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "n1"])

    def test_failover_logs_candidate_chain_and_cooling_skip(self):
        self.enable_pool()
        self.store.mark_cooldown("B_通用", "n1", "B", 120, "model")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        FakeBackend.text_failures["m2"] = lambda: HTTPError(503)

        records: list[tuple[str, str]] = []
        fake_logger = SimpleNamespace(
            info=lambda msg, *a: records.append(("info", msg % a)),
            debug=lambda msg, *a: records.append(("debug", msg % a)),
            warning=lambda msg, *a: None,
        )
        original = manager_provider.logger
        manager_provider.logger = fake_logger
        self.addCleanup(setattr, manager_provider, "logger", original)

        with self.assertRaises(HTTPError):
            asyncio.run(self.provider.text_chat(prompt="hi"))

        info_messages = [text for level, text in records if level == "info"]
        self.assertTrue(
            any("候选顺序" in m and "冷却" in m for m in info_messages),
            info_messages,
        )
        self.assertTrue(
            any("跳过冷却中的候选" in m and "剩余" in m for m in info_messages),
            info_messages,
        )

    def test_sticky_does_not_overwrite_user_session_choice(self):
        self.enable_pool(mode="sticky")
        self.store.set_conversation_override("sess-1", "B_通用/n1")
        FakeBackend.text_failures["n1"] = lambda: HTTPError(503)

        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual([call[2] for call in FakeBackend.calls], ["n1", "m2"])

        override = self.store.get_conversation_override("sess-1")
        self.assertEqual(override["model"], "n1")
        self.assertNotIn("origin", override)
        # 手动会话的降级不写入全局降级状态
        self.assertIsNone(self.store.degraded_model())

    def test_manual_session_choice_wins_over_global_degrade(self):
        self.enable_pool(mode="sticky")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "A_通用", "model": "m2"},
        )

        self.store.set_conversation_override("sess-2", "B_通用/n1")
        FakeBackend.calls = []
        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-2"))

        # 手动切换优先于全局降级，且不会清除全局降级
        self.assertEqual([call[2] for call in FakeBackend.calls], ["n1"])
        self.assertEqual(
            self.store.degraded_model(),
            {"instance_id": "A_通用", "model": "m2"},
        )

    def test_user_override_is_not_auto_released(self):
        self.enable_pool(mode="sticky")
        self.store.set_conversation_override("sess-1", "m2")

        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m2"])
        override = self.store.get_conversation_override("sess-1")
        self.assertEqual(override["model"], "m2")
        self.assertNotIn("origin", override)
        self.assertIsNone(self.store.degraded_model())

    def test_transient_does_not_write_override(self):
        self.enable_pool(mode="transient")
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        asyncio.run(self.provider.text_chat(prompt="hi", session_id="sess-1"))

        self.assertIsNone(self.store.get_conversation_override("sess-1"))
        self.assertIsNone(self.store.degraded_model())
        self.assertEqual(self.store.data["default_model"], "m1")

    def test_max_attempts_caps_calls(self):
        self.enable_pool(max_attempts=2)
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)
        FakeBackend.text_failures["m2"] = lambda: HTTPError(503)

        with self.assertRaises(HTTPError):
            asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1", "m2"])

    def test_all_cooling_fail_open_tries_once(self):
        self.enable_pool()
        self.store.mark_cooldown("A_通用", "m1", "A", 60, "source")
        self.store.mark_cooldown("A_通用", "m2", "A", 60, "source")
        self.store.mark_cooldown("B_通用", "n1", "B", 60, "source")

        asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])
        self.assertEqual(self.store.stats.record_count(), 1)

    def test_disabled_fallback_keeps_old_behavior(self):
        FakeBackend.text_failures["m1"] = lambda: HTTPError(503)

        with self.assertRaises(HTTPError):
            asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual([call[2] for call in FakeBackend.calls], ["m1"])
        self.assertFalse(self.store.is_cooling("A_通用", "m1", "A"))

    def test_fast_switch_limits_core_retries_to_one(self):
        self.enable_pool()

        asyncio.run(self.provider.text_chat(prompt="hi", request_max_retries=5))

        self.assertEqual(FakeBackend.kwargs_log[-1]["request_max_retries"], 1)

    def test_fast_switch_applies_to_stream(self):
        self.enable_pool()

        async def collect():
            return [chunk async for chunk in self.provider.text_chat_stream(
                prompt="hi", request_max_retries=5
            )]

        asyncio.run(collect())
        self.assertEqual(FakeBackend.kwargs_log[-1]["request_max_retries"], 1)

    def test_core_retries_pass_through_without_fallback(self):
        asyncio.run(self.provider.text_chat(prompt="hi", request_max_retries=5))

        self.assertEqual(FakeBackend.kwargs_log[-1]["request_max_retries"], 5)

    def test_core_retries_pass_through_with_single_candidate(self):
        self.enable_pool(targets=("B_通用/n1",))
        self.store.set_instance_enabled("B_通用", False)

        asyncio.run(self.provider.text_chat(prompt="hi", request_max_retries=5))

        self.assertEqual(FakeBackend.kwargs_log[-1]["request_max_retries"], 5)

    def test_fallback_timeout_overrides_delegated_backend(self):
        manager_provider.set_fallback_timeout(90)
        self.enable_pool()

        asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual(FakeBackend.instances[-1].provider_config["timeout"], 90)

    def test_fallback_timeout_not_applied_without_candidates(self):
        manager_provider.set_fallback_timeout(90)

        asyncio.run(self.provider.text_chat(prompt="hi"))

        self.assertEqual(FakeBackend.instances[-1].provider_config["timeout"], 120)


class ErrorClassifierTest(unittest.TestCase):
    def test_status_codes(self):
        self.assertEqual(_classify_error(HTTPError(503))[:2], (True, "model"))
        self.assertEqual(_classify_error(HTTPError(401))[:2], (True, "source"))
        self.assertEqual(_classify_error(HTTPError(404))[:2], (True, "model"))
        self.assertEqual(_classify_error(HTTPError(400))[:2], (False, ""))

    def test_message_patterns(self):
        self.assertFalse(_classify_error(ValueError("maximum context length exceeded"))[0])
        self.assertFalse(_classify_error(ValueError("content policy violation"))[0])
        self.assertEqual(
            _classify_error(ValueError("model not found"))[:2], (True, "model")
        )
        self.assertEqual(
            _classify_error(ValueError("invalid api key"))[:2], (True, "source")
        )
        self.assertEqual(
            _classify_error(TimeoutError("request timed out"))[:2], (True, "source")
        )
        self.assertFalse(_classify_error(ValueError("something odd"))[0])


if __name__ == "__main__":
    unittest.main()
