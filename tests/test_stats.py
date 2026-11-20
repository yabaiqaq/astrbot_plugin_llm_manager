"""验证 Stats 的 JSONL 追加写、旧格式迁移与跨实例同步。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

from astrbot_plugin_llm_manager import stats as stats_module  # noqa: E402
from astrbot_plugin_llm_manager.stats import Stats  # noqa: E402


class StatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "stats.json"

    def test_records_append_to_jsonl_and_aggregate(self):
        stats = Stats(self.path)
        stats.record("m1", "A_通用", 3, 2, 5)
        stats.record("m1", "A_通用", 4, 1, 5)
        stats.record("m2", "B_通用", 1, 1, 2, fallback_from="A_通用/m1")

        jsonl = self.dir / "stats.jsonl"
        self.assertTrue(jsonl.exists())
        self.assertEqual(len(jsonl.read_text(encoding="utf-8").splitlines()), 3)

        rows = {row["model"]: row for row in stats.aggregate(1)}
        self.assertEqual(rows["m1"]["calls"], 2)
        self.assertEqual(rows["m1"]["total_tokens"], 10)
        self.assertEqual(stats.fallback_calls(1), 1)
        self.assertEqual(stats.record_count(), 3)

    def test_legacy_json_is_migrated_and_kept(self):
        legacy = self.dir / "stats.json"
        legacy.write_text(
            json.dumps({
                "version": 1,
                "records": [
                    {"ts": 1, "model": "old", "instance": "A", "total_tokens": 9}
                ],
            }),
            encoding="utf-8",
        )
        stats = Stats(self.path)
        self.assertEqual(stats.record_count(), 1)
        self.assertEqual(stats.total_tokens(), 9)
        self.assertTrue(legacy.exists(), "旧 stats.json 不应被删除")
        self.assertTrue((self.dir / "stats.jsonl").exists())

    def test_refresh_sees_other_instance_appends(self):
        first = Stats(self.path)
        second = Stats(self.path)
        first.record("m1", "A", 1, 1, 2)

        self.assertEqual(second.record_count(), 1)
        second.record("m2", "B", 2, 2, 4)
        self.assertEqual(first.record_count(), 2)
        self.assertEqual(first.total_tokens(), 6)

    def test_broken_trailing_line_is_repaired(self):
        stats = Stats(self.path)
        stats.record("m1", "A", 1, 1, 2)
        jsonl = self.dir / "stats.jsonl"
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write('{"ts": 2, "model": "half"')  # 模拟崩溃留下的半行

        self.assertEqual(stats.record_count(), 1)
        stats.record("m2", "B", 1, 1, 2)
        self.assertEqual(stats.record_count(), 2)
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            json.loads(line)  # 每一行都必须是完整 JSON

    def test_record_count_capped_at_max_records(self):
        original = stats_module.MAX_RECORDS
        stats_module.MAX_RECORDS = 3
        self.addCleanup(setattr, stats_module, "MAX_RECORDS", original)
        stats = Stats(self.path)
        stats._last_prune = 0  # 强制触发压缩
        for index in range(5):
            stats.record(f"m{index}", "A", 1, 1, index + 1)

        self.assertEqual(stats.record_count(), 3)
        self.assertEqual(
            [row["model"] for row in stats.aggregate()], ["m4", "m3", "m2"]
        )


if __name__ == "__main__":
    unittest.main()
