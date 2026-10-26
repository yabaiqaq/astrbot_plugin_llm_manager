"""验证多参数指令能从原始消息中拿到完整 argv。"""
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
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_command_argv"
    )
    plugin = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMManagerPlugin"
    )
    resolver = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_and_apply"
    )
    wanted = {"llm_add", "llm_group", "llm_rm", "llm_use"}
    commands = []
    for node in plugin.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in wanted:
            node.decorator_list = []
            commands.append(node)
    module = ast.Module(body=[helper, resolver, *commands], type_ignores=[])
    namespace = {
        "AstrMessageEvent": object,
        "get_store": lambda: store,
        "logger": SimpleNamespace(warning=lambda *args: None),
        "os": __import__("os"),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class MultiArgCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.ns = load_commands(self.store)
        self.plugin = SimpleNamespace(_deny_if_not_admin=lambda event: None)

    def run_command(self, name, message):
        event = SimpleNamespace(
            message_str=message,
            unified_msg_origin="test",
            plain_result=lambda text: text,
        )

        async def collect():
            return [item async for item in self.ns[name](self.plugin, event)]

        return asyncio.run(collect())

    def test_add_receives_site_url_and_key(self):
        output, = self.run_command(
            "llm_add", "/llmm add DeepSeek https://api.deepseek.com/v1 sk-1"
        )
        self.assertIn("已添加后端源", output)
        source = self.store.find_source("DeepSeek")
        self.assertIsNotNone(source)
        self.assertEqual(source["api_base"], "https://api.deepseek.com/v1")
        self.assertEqual(source["key"], ["sk-1"])

    def test_group_add_receives_all_parts(self):
        self.store.add_source("DeepSeek", "https://api.deepseek.com/v1", "sk-1")
        output, = self.run_command(
            "llm_group", "/llmm group add DeepSeek 通用 m1 m2"
        )
        self.assertIn("已创建分组实例", output)
        inst = self.store.find_instance("DeepSeek_通用")
        self.assertIsNotNone(inst)
        self.assertEqual([m["name"] for m in inst["models"]], ["m1", "m2"])

    def test_rm_model_receives_instance_and_model(self):
        self.store.add_source("A", "https://a.example/v1", "k")
        self.store.add_instance("A", "通用", ["m1", "m2"])
        output, = self.run_command("llm_rm", "/llmm rm model A_通用 m1")
        self.assertIn("移除 1 个模型", output)
        inst = self.store.find_instance("A_通用")
        self.assertEqual([m["name"] for m in inst["models"]], ["m2"])

    def test_use_session_flag_switches_only_current_session_model(self):
        self.store.add_source("A", "https://a.example/v1", "k")
        self.store.add_instance("A", "通用", ["m1", "m2"])
        self.store.set_default("m1")
        output, = self.run_command("llm_use", "/llmm use 2 -c")
        self.assertIn("本会话已切换", output)
        self.assertEqual(self.store.data["default_model"], "m1")  # 全局默认不变
        self.assertEqual(self.store.get_conversation_override("test"), {
            "instance_id": "A_通用", "model": "m2",
        })
        self.assertEqual(self.store.resolve(umo="test")["model_name"], "m2")


if __name__ == "__main__":
    unittest.main()
