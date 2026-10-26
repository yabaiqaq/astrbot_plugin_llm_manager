"""验证 /llmm 指令参数解析（不加载 AstrBot 运行时）。"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

SOURCE = Path(__file__).resolve().parents[1] / "main.py"


def load_argv_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_command_argv"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {"AstrMessageEvent": object}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_command_argv"]


class CommandArgvTest(unittest.TestCase):
    def setUp(self):
        self.argv = load_argv_helper()

    def parse(self, message, subcommand, fallback=""):
        event = SimpleNamespace(message_str=message)
        return self.argv(event, subcommand, fallback)

    def test_add_keeps_all_arguments(self):
        self.assertEqual(
            self.parse("/llmm add DeepSeek https://api.deepseek.com/v1 sk-1", "add"),
            ["DeepSeek", "https://api.deepseek.com/v1", "sk-1"],
        )

    def test_group_subcommand_arguments(self):
        self.assertEqual(
            self.parse("/llmm group add DeepSeek 通用 m1 m2", "group"),
            ["add", "DeepSeek", "通用", "m1", "m2"],
        )

    def test_use_keeps_session_flag(self):
        self.assertEqual(self.parse("/llmm use 3 -c", "use"), ["3", "-c"])

    def test_search_keeps_multiple_keywords(self):
        self.assertEqual(self.parse("/llmm search qwen 72b", "search"), ["qwen", "72b"])

    def test_prefix_and_bot_mention(self):
        self.assertEqual(
            self.parse("@bot ／llmm rm source DeepSeek", "rm"),
            ["source", "DeepSeek"],
        )

    def test_stripped_command_prefix(self):
        self.assertEqual(
            self.parse("group add DeepSeek 推理 m1", "group"),
            ["add", "DeepSeek", "推理", "m1"],
        )

    def test_fallback_for_direct_call(self):
        event = SimpleNamespace()  # 测试或旧调用方没有 message_str
        self.assertEqual(self.argv(event, "search", "qwen 72b"), ["qwen", "72b"])

    def test_unrelated_message_falls_back(self):
        self.assertEqual(
            self.parse("普通聊天内容", "add", "DeepSeek https://a/v1 sk"),
            ["DeepSeek", "https://a/v1", "sk"],
        )


if __name__ == "__main__":
    unittest.main()
