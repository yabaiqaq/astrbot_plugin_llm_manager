"""验证 /llmm fallback 指令的参数解析、权限与输出。"""
from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

from astrbot_plugin_llm_manager.store import Store  # noqa: E402

SOURCE = Path(__file__).resolve().parents[1] / "main.py"


def load_commands(store):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted_functions = {
        "_command_argv", "_fallback_usage", "_parse_pool_indexes",
        "_render_fallback", "_modality_mark",
    }
    helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions
    ]
    plugin = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMManagerPlugin"
    )
    command = next(
        node for node in plugin.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "llm_fallback"
    )
    command.decorator_list = []
    module = ast.Module(body=[*helpers, command], type_ignores=[])
    namespace = {
        "AstrMessageEvent": object,
        "Store": Store,
        "get_store": lambda: store,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class FallbackCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.store.add_source("A", "https://a.example/v1", "ka", stype="fake")
        self.store.add_instance("A", "通用", ["m1", "m2", "m3"])
        self.store.add_source("B", "https://b.example/v1", "kb", stype="fake")
        self.store.add_instance("B", "通用", ["n1"])
        self.store.set_default("m1")
        self.ns = load_commands(self.store)
        self.is_admin = True
        self.plugin = SimpleNamespace(
            _deny_if_not_admin=lambda event: None,
            _is_bot_admin=lambda event: self.is_admin,
        )

    def run_command(self, message):
        event = SimpleNamespace(
            message_str=message,
            unified_msg_origin="test",
            plain_result=lambda text: text,
        )

        async def collect():
            return [item async for item in self.ns["llm_fallback"](self.plugin, event)]

        return asyncio.run(collect())

    def test_status_before_configuration(self):
        output, = self.run_command("/llmm fallback")
        self.assertIn("故障转移：关闭", output)
        self.assertIn("模式：transient", output)
        self.assertIn("备选池为空", output)

    def test_add_targets_and_enable(self):
        output, = self.run_command("/llmm fallback add 2 3")
        self.assertIn("已加入 2 条备选", output)
        self.assertIn("1. A_通用/m2", output)
        self.assertIn("2. A_通用/m3", output)
        self.assertEqual(
            [e["model"] for e in self.store.fallback_config()["pool"]], ["m2", "m3"]
        )

        output, = self.run_command("/llmm fallback on")
        self.assertIn("故障转移已开启", output)
        self.assertTrue(self.store.fallback_config()["enabled"])

    def test_enable_empty_pool_rejected(self):
        output, = self.run_command("/llmm fallback on")
        self.assertIn("备选池为空，无法开启", output)
        self.assertFalse(self.store.fallback_config()["enabled"])

    def test_rm_supports_ranges(self):
        self.run_command("/llmm fallback add 1 2 3")
        output, = self.run_command("/llmm fallback rm 1-2")
        self.assertIn("已移除 2 条", output)
        self.assertEqual(
            [e["model"] for e in self.store.fallback_config()["pool"]], ["m3"]
        )

    def test_move_changes_priority(self):
        self.run_command("/llmm fallback add 1 2 3")
        output, = self.run_command("/llmm fallback move 3 1")
        self.assertIn("已把第 3 条移动到第 1 位", output)
        self.assertEqual(
            [e["model"] for e in self.store.fallback_config()["pool"]],
            ["m3", "m1", "m2"],
        )

    def test_mode_max_and_cooldown(self):
        output, = self.run_command("/llmm fallback mode sticky")
        self.assertIn("已切换为 sticky", output)
        self.assertEqual(self.store.fallback_config()["mode"], "sticky")

        output, = self.run_command("/llmm fallback max 4")
        self.assertIn("尝试上限已设为 4", output)
        self.assertEqual(self.store.fallback_config()["max_attempts"], 4)

        output, = self.run_command("/llmm fallback cooldown 0")
        self.assertIn("冷却已设为 0 秒", output)
        self.assertEqual(self.store.fallback_config()["cooldown_seconds"], 0)

    def test_clear_requires_confirmation(self):
        self.run_command("/llmm fallback add 1 2")
        self.run_command("/llmm fallback on")

        output, = self.run_command("/llmm fallback clear")
        self.assertIn("确认请执行", output)
        self.assertEqual(len(self.store.fallback_config()["pool"]), 2)

        output, = self.run_command("/llmm fallback clear confirm")
        self.assertIn("已清空 2 条备选", output)
        self.assertEqual(self.store.fallback_config()["pool"], [])
        self.assertFalse(self.store.fallback_config()["enabled"])

    def test_check_reports_missing_entries(self):
        self.run_command("/llmm fallback add 1 3")
        self.store.remove_models("A_通用", ["m3"])
        output, = self.run_command("/llmm fallback check")
        self.assertIn("可解析 1 条", output)
        self.assertIn("失效 1 条", output)
        self.assertIn("实例或模型已删除", output)

    def test_non_admin_can_view_but_cannot_write(self):
        self.is_admin = False
        output, = self.run_command("/llmm fallback")
        self.assertIn("故障转移：关闭", output)

        output, = self.run_command("/llmm fallback add 1")
        self.assertIn("此操作仅管理员可用", output)
        self.assertEqual(self.store.fallback_config()["pool"], [])

    def test_unknown_action_shows_usage(self):
        output, = self.run_command("/llmm fallback wat")
        self.assertIn("用法：", output)
        self.assertIn("/llmm fallback add", output)


if __name__ == "__main__":
    unittest.main()
