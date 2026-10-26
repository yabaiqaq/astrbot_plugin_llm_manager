"""备选池数据模型、解析与运行时冷却测试。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

from astrbot_plugin_llm_manager.store import (  # noqa: E402
    MAX_FALLBACK_POOL,
    STORE_VERSION,
    Store,
)


class FallbackStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "backends.json"
        self.store = Store(self.path)
        self.store.add_source("A", "https://a.example/v1", "ka", stype="fake")
        self.store.add_instance("A", "通用", ["m1", "m2", "m3"])
        self.store.add_source("B", "https://b.example/v1", "kb", stype="fake")
        self.store.add_instance("B", "通用", ["n1"])

    def test_defaults_are_disabled_and_safe(self):
        fb = self.store.fallback_config()
        self.assertFalse(fb["enabled"])
        self.assertEqual(fb["mode"], "transient")
        self.assertEqual(fb["max_attempts"], 3)
        self.assertEqual(fb["cooldown_seconds"], 120)
        self.assertEqual(fb["pool"], [])
        self.assertEqual(self.store.data["version"], STORE_VERSION)

    def test_normalize_invalid_values_and_duplicate_pool(self):
        self.path.write_text(json.dumps({
            "version": 1,
            "fallback": {
                "enabled": True,
                "mode": "weird",
                "max_attempts": 99,
                "cooldown_seconds": -5,
                "pool": [
                    {"instance_id": "A_通用", "model": "m1"},
                    {"instance_id": "A_通用", "model": "m1"},
                    "bad",
                    {"instance_id": "", "model": "x"},
                ],
            },
        }), encoding="utf-8")
        store = Store(self.path)
        fb = store.fallback_config()
        self.assertTrue(fb["enabled"])
        self.assertEqual(fb["mode"], "transient")
        self.assertEqual(fb["max_attempts"], 5)
        self.assertEqual(fb["cooldown_seconds"], 0)
        self.assertEqual(fb["pool"], [{"instance_id": "A_通用", "model": "m1"}])
        self.assertEqual(store.data["version"], STORE_VERSION)

    def test_add_resolves_numbers_paths_and_skips_duplicates(self):
        result = self.store.add_fallback_targets(["2", "B_通用/n1", "4", "nope"])
        self.assertEqual(
            [item["model"] for item in result["added"]], ["m2", "n1"]
        )
        self.assertEqual(result["duplicates"], ["4"])  # 序号 4 也是 n1
        self.assertEqual(result["not_found"], ["nope"])
        self.assertEqual(result["start_index"], 1)
        self.assertEqual(
            self.store.fallback_config()["pool"],
            [
                {"instance_id": "A_通用", "model": "m2"},
                {"instance_id": "B_通用", "model": "n1"},
            ],
        )

    def test_add_honors_pool_limit(self):
        models = [f"x{i}" for i in range(MAX_FALLBACK_POOL + 2)]
        self.store.add_models("A_通用", models)
        result = self.store.add_fallback_targets(models)
        self.assertEqual(len(result["added"]), MAX_FALLBACK_POOL)
        self.assertEqual(len(result["overflow"]), 2)
        self.assertEqual(len(self.store.fallback_config()["pool"]), MAX_FALLBACK_POOL)

    def test_enable_requires_non_empty_pool(self):
        self.assertFalse(self.store.set_fallback_enabled(True))
        self.store.add_fallback_targets(["1"])
        self.assertTrue(self.store.set_fallback_enabled(True))
        self.assertTrue(self.store.fallback_config()["enabled"])

    def test_remove_moves_and_auto_disables_when_empty(self):
        self.store.add_fallback_targets(["1", "2", "3"])
        self.store.set_fallback_enabled(True)
        self.assertTrue(self.store.move_fallback(3, 1))
        self.assertEqual(
            [e["model"] for e in self.store.fallback_config()["pool"]],
            ["m3", "m1", "m2"],
        )
        result = self.store.remove_fallback_indexes([1, 2])
        self.assertEqual([r["index"] for r in result["removed"]], [1, 2])
        self.assertEqual(result["pool_size"], 1)
        self.assertTrue(result["enabled"])
        result = self.store.remove_fallback_indexes([1])
        self.assertEqual(result["pool_size"], 0)
        self.assertTrue(result["auto_disabled"])
        self.assertFalse(self.store.fallback_config()["enabled"])

    def test_remove_invalid_indexes(self):
        self.store.add_fallback_targets(["1"])
        result = self.store.remove_fallback_indexes([0, 2])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["invalid"], [0, 2])

    def test_clear_disables_and_returns_count(self):
        self.store.add_fallback_targets(["1", "2"])
        self.store.set_fallback_enabled(True)
        self.assertEqual(self.store.clear_fallback(), 2)
        self.assertEqual(self.store.fallback_config()["pool"], [])
        self.assertFalse(self.store.fallback_config()["enabled"])
        self.assertEqual(self.store.clear_fallback(), 0)

    def test_resolve_skips_missing_disabled_duplicate_and_modalities(self):
        self.store.add_fallback_targets(["1", "2", "3", "4"])
        # m2 改成纯文本，m3 从目录删除
        inst = self.store.find_instance("A_通用")
        for model in inst["models"]:
            if model["name"] == "m2":
                model["modalities"] = ["text"]
        self.store.remove_models("A_通用", ["m3"])
        self.store.save()

        routed = self.store.resolve_fallback_entries()
        self.assertEqual([r["model_name"] for r in routed], ["m1", "m2", "n1"])

        only_image = self.store.resolve_fallback_entries({"image"})
        self.assertEqual(only_image, [])

        only_text = self.store.resolve_fallback_entries({"text"})
        self.assertEqual([r["model_name"] for r in only_text], ["m1", "m2", "n1"])

    def test_fallback_status_reports_missing_and_disabled(self):
        self.store.add_fallback_targets(["1", "2", "4"])
        self.store.remove_models("A_通用", ["m2"])
        self.store.set_instance_enabled("B_通用", False)
        statuses = self.store.fallback_status()
        self.assertEqual([s["status"] for s in statuses], ["ok", "missing", "disabled"])
        self.assertEqual(statuses[0]["index"], 1)
        self.assertEqual(statuses[0]["num"], 1)

    def test_cooldown_model_and_source_scope(self):
        self.store.mark_cooldown("A_通用", "m1", "A", 60, "model")
        self.assertGreater(self.store.cooldown_remaining("A_通用", "m1", "A"), 0)
        self.assertEqual(self.store.cooldown_remaining("A_通用", "m2", "A"), 0)
        self.store.clear_cooldown("A_通用", "m1", "A")
        self.assertEqual(self.store.cooldown_remaining("A_通用", "m1", "A"), 0)

        self.store.mark_cooldown("A_通用", "m1", "A", 60, "source")
        self.assertGreater(self.store.cooldown_remaining("A_通用", "m2", "A"), 0)
        self.assertGreater(self.store.cooldown_remaining("A_通用", "m3", "A"), 0)
        self.assertEqual(self.store.cooldown_remaining("B_通用", "n1", "B"), 0)
        self.store.reset_cooldowns()
        self.assertFalse(self.store.is_cooling("A_通用", "m1", "A"))

    def test_zero_cooldown_does_nothing(self):
        self.store.mark_cooldown("A_通用", "m1", "A", 0, "source")
        self.assertFalse(self.store.is_cooling("A_通用", "m1", "A"))


if __name__ == "__main__":
    unittest.main()
