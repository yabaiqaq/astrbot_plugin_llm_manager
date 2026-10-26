"""独立验证搜索指令的输出与权限，不加载 AstrBot 或注册 Provider。"""
from __future__ import annotations

import ast
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from unittest.mock import Mock

import _astrbot_stub  # noqa: E402,F401  无 astrbot 环境时注入 logger 替身
from astrbot_plugin_llm_manager.store import DEFAULT_MODALITIES, Store


def load_search_command(store, render):
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_modality_mark")
    argv = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_command_argv")
    plugin = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LLMManagerPlugin")
    command = next(node for node in plugin.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "llm_search")
    command.decorator_list = []
    module = ast.Module(body=[helper, argv, command], type_ignores=[])
    namespace = {
        "AstrMessageEvent": object,
        "DEFAULT_MODALITIES": DEFAULT_MODALITIES,
        "get_store": lambda: store,
        "render_catalog_image": render,
        "logger": SimpleNamespace(warning=lambda *args: None),
        "os": os,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["llm_search"]


class SearchCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.store.add_source("站点", "https://example.com/v1", "test-key")
        self.store.add_instance("站点", "通用", ["other", "Qwen/Qwen2.5-72B"])
        self.store.set_default("2")
        self.image_path = Path(self.tmp.name) / "result.png"
        self.render = Mock(return_value=str(self.image_path))
        self.command = load_search_command(self.store, self.render)
        self.event = SimpleNamespace(
            unified_msg_origin="test-session", plain_result=lambda text: text,
            image_result=self.image_result,
        )
        self.plugin = SimpleNamespace(_deny_if_not_admin=lambda event: None)

    def run_command(self, query):
        async def collect():
            return [result async for result in self.command(self.plugin, self.event, query)]
        return asyncio.run(collect())

    def image_result(self, path):
        self.assertTrue(Path(path).exists())
        return ("image", path)

    def test_search_returns_filtered_image_and_cleans_tempfile(self):
        self.image_path.write_bytes(b"test-image")
        self.assertEqual(self.run_command("qwen 72b"), [("image", str(self.image_path))])
        self.render.assert_called_once_with(
            self.store, "test-session", catalog=self.store.search_models("qwen 72b"),
        )
        self.assertFalse(self.image_path.exists())

    def test_render_failure_falls_back_to_filtered_text(self):
        self.render.side_effect = RuntimeError("render error")
        output, = self.run_command("qwen 72b")
        self.assertIn("共 1 条匹配", output)
        self.assertIn("[2] Qwen/Qwen2.5-72B", output)
        self.assertIn("站点：站点 · 分组：站点_通用", output)
        self.assertIn("使用中", output)
        self.assertNotIn("test-key", output)
        self.store.set_instance_enabled("站点_通用", False)
        output, = self.run_command("qwen")
        self.assertIn("停用", output)
        self.assertNotIn("使用中", output)

    def test_empty_query(self):
        output, = self.run_command("   ")
        self.assertIn("用法：/llmm search", output)

    def test_no_match(self):
        output, = self.run_command("nonexistent")
        self.assertIn("未找到模型名匹配", output)

    def test_permission_denied(self):
        self.plugin._deny_if_not_admin = lambda event: "拒绝访问"
        self.assertEqual(self.run_command("qwen"), ["拒绝访问"])


if __name__ == "__main__":
    unittest.main()
