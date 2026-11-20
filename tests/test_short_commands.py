"""验证 /llmm 短名指令能正确路由到现有实现。"""
from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _astrbot_stub  # noqa: E402,F401

from astrbot_plugin_llm_manager.store import Store  # noqa: E402

SOURCE = Path(__file__).resolve().parents[1] / "main.py"

MODULE_HELPERS = {
    "_command_argv", "_alias_argv", "_fallback_usage", "_short_help",
    "_setup_help", "_full_help", "_parse_pool_indexes", "_render_fallback",
    "_modality_mark", "_apply_fallback_config", "_store_api_problems",
    "_resolve_and_apply", "_render_tree",
}

COMMANDS = {
    "llm_list", "llm_ls", "llm_l_short",
    "llm_search", "llm_s",
    "llm_use", "llm_u",
    "llm_default", "llm_d",
    "llm_fallback", "llm_f", "llm_ft", "llm_fc",
    "llm_stats", "llm_st",
    "llm_test", "llm_t",
    "llm_help", "llm_h",
}


def load(store):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in MODULE_HELPERS
    ]
    plugin_cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMManagerPlugin"
    )
    commands = []
    for node in plugin_cls.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name in COMMANDS:
            node.decorator_list = []
            commands.append(node)
    module = ast.Module(body=[*helpers, *commands], type_ignores=[])
    namespace = {
        "AstrMessageEvent": object,
        "Store": Store,
        "asyncio": asyncio,
        "get_store": lambda: store,
        "logger": SimpleNamespace(
            warning=lambda *args: None,
            info=lambda *args: None,
            debug=lambda *args: None,
        ),
        "os": __import__("os"),
        "render_catalog_image": None,
        "REQUIRED_STORE_API_VERSION": 1,
        "_REQUIRED_STORE_METHODS": (
            "degraded_model",
            "set_degraded",
            "clear_degraded",
            "route_entry",
            "refresh_if_changed",
        ),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class ShortCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "backends.json")
        self.store.add_source("A", "https://a.example/v1", "ka", stype="fake")
        self.store.add_instance("A", "通用", ["m1", "m2", "m3"])
        self.store.add_source("B", "https://b.example/v1", "kb", stype="fake")
        self.store.add_instance("B", "通用", ["n1"])
        self.store.set_default("m1")

        namespace = load(self.store)
        self.ns = namespace

        class Plugin:
            pass

        self.plugin = Plugin()
        for name, fn in namespace.items():
            if name.startswith("llm_"):
                setattr(self.plugin, name, MethodType(fn, self.plugin))
        self.plugin._deny_if_not_admin = lambda event: None
        self.plugin._is_bot_admin = lambda event: True

    def event(self, message):
        return SimpleNamespace(
            message_str=message,
            unified_msg_origin="test",
            plain_result=lambda text: text,
        )

    def run_cmd(self, name, message):
        async def collect():
            return [
                item async for item
                in getattr(self.plugin, name)(self.event(message))
            ]

        return asyncio.run(collect())

    def fake_test(self):
        tested = []

        async def _test(store, item):
            tested.append(item["model"])
            return (item, True, 88.0, None)

        self.plugin._test_one_model = _test
        return tested

    # ---------- 参数解析 ----------

    def test_alias_argv_reads_message_after_short_name(self):
        parse = self.ns["_alias_argv"]
        self.assertEqual(
            parse(self.event("/llmm f cd 60"), ("f", "fallback")), ["cd", "60"]
        )
        self.assertEqual(parse(self.event("/llmm ft"), ("ft",)), [])
        self.assertEqual(
            parse(self.event("/llmm u 3 -c"), ("u", "use")), ["3", "-c"]
        )
        # 调用方已剥掉 /llmm 前缀时也能解析
        self.assertEqual(
            parse(self.event("f cd 60"), ("f", "fallback")), ["cd", "60"]
        )

    # ---------- 查看 / 切换 ----------

    def test_ls_routes_to_list_text(self):
        output, = self.run_cmd("llm_ls", "/llmm ls -t")
        self.assertIn("—— 共", output)
        self.assertIn("[1] m1", output)

    def test_l_alias_routes_to_list(self):
        output, = self.run_cmd("llm_l_short", "/llmm l -t")
        self.assertIn("—— 共", output)

    def test_s_routes_to_search(self):
        output, = self.run_cmd("llm_s", "/llmm s m1")
        self.assertIn("模型搜索", output)
        self.assertIn("m1", output)

    def test_u_routes_to_use(self):
        output, = self.run_cmd("llm_u", "/llmm u 2")
        self.assertIn("已全局切换", output)
        self.assertEqual(self.store.data["default_model"], "m2")

    def test_u_session_flag(self):
        output, = self.run_cmd("llm_u", "/llmm u 3 -c")
        self.assertIn("本会话已切换", output)
        self.assertEqual(self.store.data["default_model"], "m1")
        self.assertEqual(
            self.store.get_conversation_override("test"),
            {"instance_id": "A_通用", "model": "m3"},
        )

    def test_d_routes_to_default(self):
        output, = self.run_cmd("llm_d", "/llmm d 3")
        self.assertIn("全局默认已设为", output)
        self.assertEqual(self.store.data["default_model"], "m3")

    # ---------- 测速 / 统计 ----------

    def test_t_tests_current_model(self):
        self.fake_test()
        output, = self.run_cmd("llm_t", "/llmm t")
        self.assertIn("连通正常", output)

    def test_t_passes_multiple_targets(self):
        tested = self.fake_test()
        output, = self.run_cmd("llm_t", "/llmm t 2 3")
        self.assertIn("2/2 成功", output)
        self.assertEqual(sorted(tested), ["m2", "m3"])

    def test_ft_runs_fallback_latency_test(self):
        self.run_cmd("llm_f", "/llmm f add 2 3")
        tested = self.fake_test()
        output, = self.run_cmd("llm_ft", "/llmm ft")
        self.assertIn("故障转移延迟测试", output)
        self.assertIn("3/3 成功", output)
        self.assertEqual(sorted(tested), ["m1", "m2", "m3"])

    def test_st_routes_to_stats(self):
        output, = self.run_cmd("llm_st", "/llmm st 1d")
        self.assertIn("暂无 Token 消耗记录", output)

    # ---------- 故障转移 ----------

    def test_f_status(self):
        output, = self.run_cmd("llm_f", "/llmm f")
        self.assertIn("故障转移：关闭", output)

    def test_f_cd_maps_to_cooldown(self):
        output, = self.run_cmd("llm_f", "/llmm f cd 120")
        self.assertIn("冷却已设为 120 秒", output)
        self.assertEqual(self.store.fallback_config()["cooldown_seconds"], 120)

    def test_f_mv_maps_to_move(self):
        self.run_cmd("llm_f", "/llmm f add 1 2 3")
        output, = self.run_cmd("llm_f", "/llmm f mv 3 1")
        self.assertIn("已把第 3 条移动到第 1 位", output)
        self.assertEqual(
            [entry["model"] for entry in self.store.fallback_config()["pool"]],
            ["m3", "m1", "m2"],
        )

    def test_fc_routes_to_check(self):
        self.run_cmd("llm_f", "/llmm f add 1 2")
        output, = self.run_cmd("llm_fc", "/llmm fc")
        self.assertIn("备选池检查", output)
        self.assertIn("可解析 2 条", output)

    def test_f_write_action_still_requires_admin(self):
        self.plugin._is_bot_admin = lambda event: False
        output, = self.run_cmd("llm_f", "/llmm f on")
        self.assertIn("此操作仅管理员可用", output)
        self.assertFalse(self.store.fallback_config()["enabled"])

    # ---------- 帮助 ----------

    def test_h_shows_short_commands(self):
        output, = self.run_cmd("llm_h", "/llmm h")
        self.assertIn("/llmm ft", output)
        self.assertIn("/llmm h fb", output)
        self.assertIn("短名与全称等价", output)

    def test_h_topics(self):
        output, = self.run_cmd("llm_h", "/llmm h fb")
        self.assertIn("/llmm f add", output)

        output, = self.run_cmd("llm_h", "/llmm h setup")
        self.assertIn("/llmm import", output)

        output, = self.run_cmd("llm_h", "/llmm h full")
        self.assertIn("/llmm ls | list", output)


if __name__ == "__main__":
    unittest.main()