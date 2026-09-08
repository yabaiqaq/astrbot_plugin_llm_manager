"""store.py 的独立单元测试（不依赖 AstrBot 运行时）。

运行方式：
    python -m unittest discover -s tests -p "test_*.py"
或直接：
    python tests/test_store.py
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from store import DEFAULT_MODALITIES, Store


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "backends.json"
        self.store = Store(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def seed(self) -> Store:
        s = self.store
        s.add_source("DeepSeek", "https://api.deepseek.com/v1", "sk-a")
        s.add_instance("DeepSeek", "通用", ["deepseek-chat", "deepseek-v3"])
        s.add_instance("DeepSeek", "推理", ["deepseek-reasoner"])
        s.add_source("硅基流动", "https://api.siliconflow.cn/v1", "sk-b")
        s.add_instance("硅基流动", "通用", ["Qwen/Qwen2.5-72B-Instruct"])
        return s

    # ---------- 一级：Source ----------

    def test_add_source_reuse_same_site_and_base(self):
        s = self.store
        src1 = s.add_source("DeepSeek", "https://api.deepseek.com/v1", "sk-a")
        src2 = s.add_source("DeepSeek", "https://api.deepseek.com/v1", "sk-b")
        self.assertEqual(src1["id"], src2["id"])
        self.assertEqual(len(src1["key"]), 2)  # key 合并且去重
        self.assertEqual(len(s.sources()), 1)

    def test_add_source_same_site_diff_base(self):
        s = self.store
        s.add_source("OpenAI", "https://api.openai.com/v1", "k1")
        s.add_source("OpenAI", "https://ark.cn-beijing.volces.com/api/v3", "k2")
        self.assertEqual(len(s.sources()), 2)
        self.assertEqual(s.sources()[1]["id"], "OpenAI-2")

    def test_source_config_shape(self):
        s = self.store
        src = s.add_source("DeepSeek", "https://api.deepseek.com/v1/", "sk-a")
        cfg = s.source_config(src)
        self.assertEqual(cfg["api_base"], "https://api.deepseek.com/v1")  # 去尾部斜杠
        self.assertEqual(cfg["key"], ["sk-a"])
        self.assertIn("type", cfg)
        self.assertIn("provider_type", cfg)

    # ---------- 二级：Instance ----------

    def test_instance_id_auto_and_collision(self):
        s = self.seed()
        inst = s.find_instance("DeepSeek_推理")
        self.assertIsNotNone(inst)
        # 撞名自动加后缀
        s.add_instance("DeepSeek", "推理", ["m2"])
        self.assertIsNotNone(s.find_instance("DeepSeek_推理-2"))

    def test_add_instance_unknown_site_raises(self):
        with self.assertRaises(ValueError):
            self.store.add_instance("不存在", "通用", ["m"])

    def test_remove_instance_clears_references(self):
        s = self.seed()
        s.set_default("DeepSeek_推理")
        s.set_conversation_override("umo-1", "DeepSeek_推理")
        self.assertTrue(s.remove_instance("DeepSeek_推理"))
        self.assertEqual(s.data["default_instance"], "")
        self.assertNotIn("umo-1", s.data["overrides"])

    # ---------- 三级：Model ----------

    def test_add_models_dedup(self):
        s = self.seed()
        n = s.add_models("DeepSeek_通用", ["deepseek-chat", "deepseek-r1"])
        self.assertEqual(n, 1)  # deepseek-chat 已存在
        inst = s.find_instance("DeepSeek_通用")
        self.assertEqual(len(inst["models"]), 3)

    def test_remove_models(self):
        s = self.seed()
        n = s.remove_models("DeepSeek_通用", ["deepseek-chat"])
        self.assertEqual(n, 1)

    # ---------- 目录与编号 ----------

    def test_catalog_numbering(self):
        s = self.seed()
        cat = s.build_catalog()
        self.assertEqual([i["num"] for i in cat], [1, 2, 3, 4])
        self.assertEqual(cat[0]["model"], "deepseek-chat")
        self.assertEqual(cat[3]["model"], "Qwen/Qwen2.5-72B-Instruct")

    def test_find_in_catalog_by_num_model_instance(self):
        s = self.seed()
        self.assertEqual(s.find_in_catalog("3")["model"], "deepseek-reasoner")
        self.assertEqual(s.find_in_catalog("deepseek-v3")["num"], 2)
        self.assertEqual(s.find_in_catalog("DeepSeek_推理")["instance_id"], "DeepSeek_推理")

    def test_find_in_catalog_by_full_path(self):
        s = self.seed()
        item = s.find_in_catalog("DeepSeek_通用/deepseek-v3")
        self.assertIsNotNone(item)
        self.assertEqual(item["num"], 2)
        self.assertEqual(item["model"], "deepseek-v3")
        self.assertEqual(item["instance_id"], "DeepSeek_通用")
        # 不存在的路径
        self.assertIsNone(s.find_in_catalog("DeepSeek_通用/nonexistent"))
        self.assertIsNone(s.find_in_catalog("Nonexistent_组/deepseek-chat"))

    # ---------- 路由 ----------

    def test_resolve_req_model(self):
        s = self.seed()
        r = s.resolve(req_model="deepseek-reasoner")
        self.assertEqual(r["model_name"], "deepseek-reasoner")
        self.assertEqual(r["instance"]["id"], "DeepSeek_推理")

    def test_resolve_conversation_override(self):
        s = self.seed()
        s.set_conversation_override("umo-A", "硅基流动_通用")
        r = s.resolve(umo="umo-A")
        self.assertEqual(r["instance"]["id"], "硅基流动_通用")

    def test_resolve_default_then_fallback(self):
        s = self.seed()
        r = s.resolve()
        self.assertEqual(r["instance"]["id"], "DeepSeek_通用")  # 第一个可用实例
        s.set_default("DeepSeek_推理")
        r = s.resolve()
        self.assertEqual(r["model_name"], "deepseek-reasoner")

    def test_resolve_respects_disabled(self):
        s = self.seed()
        s.set_instance_enabled("DeepSeek_通用", False)
        r = s.resolve()
        self.assertEqual(r["instance"]["id"], "DeepSeek_推理")

    def test_resolve_empty_returns_none(self):
        r = Store(self.path).resolve()
        self.assertIsNone(r)

    def test_persistence(self):
        s = self.seed()
        s.set_default("DeepSeek_推理")
        s2 = Store(self.path)  # 重新读取
        self.assertEqual(s2.data["default_instance"], "DeepSeek_推理")
        self.assertEqual(len(s2.sources()), 2)

    def test_modalities_default(self):
        s = self.seed()
        item = s.find_in_catalog("deepseek-chat")
        self.assertEqual(item["modalities"], DEFAULT_MODALITIES)

    # ---------- 从 AstrBot 配置导入 ----------

    def test_import_from_astrbot_config(self):
        s = Store(self.path)
        config = {
            "provider_sources": [
                {
                    "id": "deepseek",
                    "provider": "deepseek",
                    "type": "openai_chat_completion",
                    "provider_type": "chat_completion",
                    "key": ["sk-test"],
                    "api_base": "https://api.deepseek.com/v1/",
                    "timeout": 120,
                    "proxy": "",
                    "custom_headers": {},
                    "enable": True,
                },
                {
                    "id": "openai_1",
                    "provider": "openai",
                    "type": "openai_chat_completion",
                    "provider_type": "chat_completion",
                    "key": ["sk-other"],
                    "api_base": "https://api.apilio.cn/v1",
                    "timeout": 120,
                    "proxy": "",
                    "custom_headers": {},
                    "enable": True,
                },
            ],
            "provider": [
                {
                    "id": "deepseek/deepseek-v4-flash",
                    "enable": True,
                    "provider_source_id": "deepseek",
                    "model": "deepseek-v4-flash",
                    "modalities": ["text", "tool_use"],
                },
                {
                    "id": "deepseek/deepseek-v4-pro",
                    "enable": True,
                    "provider_source_id": "deepseek",
                    "model": "deepseek-v4-pro",
                    "modalities": ["text", "tool_use"],
                },
                {
                    "id": "openai_1/gpt-5.5",
                    "enable": True,
                    "provider_source_id": "openai_1",
                    "model": "gpt-5.5",
                    "modalities": ["text", "tool_use", "image"],
                },
                {
                    # dashscope agent_runner，没有 provider_source_id，应跳过
                    "id": "dashscope",
                    "type": "dashscope",
                    "provider_type": "agent_runner",
                    "enable": True,
                },
            ],
        }
        result = s.import_from_astrbot_config(config)
        self.assertEqual(result["imported_sources"], 2)
        self.assertEqual(result["imported_instances"], 2)
        self.assertEqual(result["imported_models"], 3)
        self.assertEqual(result["skipped"], ["dashscope"])
        self.assertEqual(result["total_sources"], 2)
        self.assertEqual(result["total_models"], 3)

        # 验证三级结构
        cat = s.build_catalog()
        self.assertEqual([i["model"] for i in cat], [
            "deepseek-v4-flash", "deepseek-v4-pro", "gpt-5.5"
        ])
        # 实例 id 就是提供商源唯一 ID
        self.assertEqual(cat[0]["instance_id"], "deepseek")
        self.assertEqual(cat[2]["instance_id"], "openai_1")
        # api_base 去尾部斜杠
        src = s.find_source("deepseek")
        self.assertEqual(src["api_base"], "https://api.deepseek.com/v1")
        # modalities 保留
        self.assertEqual(cat[2]["modalities"], ["text", "tool_use", "image"])

    def test_import_idempotent(self):
        s = Store(self.path)
        config = {
            "provider_sources": [
                {
                    "id": "deepseek", "provider": "deepseek",
                    "type": "openai_chat_completion", "provider_type": "chat_completion",
                    "key": ["sk-test"], "api_base": "https://api.deepseek.com/v1",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
            ],
            "provider": [
                {"id": "deepseek/m1", "enable": True, "provider_source_id": "deepseek",
                 "model": "m1", "modalities": ["text"]},
            ],
        }
        r1 = s.import_from_astrbot_config(config)
        r2 = s.import_from_astrbot_config(config)
        self.assertEqual(r1["imported_sources"], 1)
        self.assertEqual(r1["imported_models"], 1)
        # 第二次导入不应重复创建
        self.assertEqual(r2["imported_sources"], 0)
        self.assertEqual(r2["imported_instances"], 0)
        self.assertEqual(r2["imported_models"], 0)
        self.assertEqual(r2["total_models"], 1)

    def test_import_preserves_original_default(self):
        """导入时自动读取原 default_provider_id 并设为全局默认。"""
        s = Store(self.path)
        config = {
            "provider_sources": [
                {
                    "id": "deepseek", "provider": "deepseek",
                    "type": "openai_chat_completion", "provider_type": "chat_completion",
                    "key": ["sk-a"], "api_base": "https://api.deepseek.com/v1",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
                {
                    "id": "gemini", "provider": "gemini",
                    "type": "openai_chat_completion", "provider_type": "chat_completion",
                    "key": ["sk-b"], "api_base": "https://generativelanguage.googleapis.com/v1beta",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
            ],
            "provider": [
                {"id": "deepseek/m1", "enable": True, "provider_source_id": "deepseek",
                 "model": "m1", "modalities": ["text"]},
                {"id": "gemini/gemini-2.5-pro", "enable": True, "provider_source_id": "gemini",
                 "model": "gemini-2.5-pro", "modalities": ["text"]},
            ],
            "provider_settings": {
                "default_provider_id": "gemini/gemini-2.5-pro",
            },
        }
        result = s.import_from_astrbot_config(config)
        self.assertTrue(result["auto_default_set"])
        self.assertEqual(result["original_default"], "gemini/gemini-2.5-pro")
        self.assertEqual(s.data["default_instance"], "gemini")
        self.assertEqual(s.data["default_model"], "gemini-2.5-pro")
        # resolve 应返回 gemini 而不是兜底的 deepseek
        routed = s.resolve()
        self.assertEqual(routed["model_name"], "gemini-2.5-pro")

    def test_import_does_not_override_existing_default(self):
        """如果用户已设全局默认，导入时不覆盖。"""
        s = Store(self.path)
        config = {
            "provider_sources": [
                {
                    "id": "deepseek", "provider": "deepseek",
                    "type": "openai_chat_completion", "provider_type": "chat_completion",
                    "key": ["sk-a"], "api_base": "https://api.deepseek.com/v1",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
            ],
            "provider": [
                {"id": "deepseek/m1", "enable": True, "provider_source_id": "deepseek",
                 "model": "m1", "modalities": ["text"]},
            ],
            "provider_settings": {
                "default_provider_id": "deepseek/m1",
            },
        }
        # 先手动设一个默认
        s.add_source("Other", "https://other.com/v1", "sk-other")
        s.add_instance("Other", "默认", ["other-model"])
        s.set_default("other-model")
        self.assertEqual(s.data["default_model"], "other-model")

        result = s.import_from_astrbot_config(config)
        self.assertFalse(result["auto_default_set"])
        # 原有默认不被覆盖
        self.assertEqual(s.data["default_model"], "other-model")

    def test_import_skips_llm_manager_self_as_default(self):
        """如果原 default_provider_id 指向 llm_manager 自身，不自动设置。"""
        s = Store(self.path)
        config = {
            "provider_sources": [
                {
                    "id": "llm_manager", "provider": "llm_manager",
                    "type": "llm_manager", "provider_type": "chat_completion",
                    "key": [], "api_base": "",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
                {
                    "id": "deepseek", "provider": "deepseek",
                    "type": "openai_chat_completion", "provider_type": "chat_completion",
                    "key": ["sk-a"], "api_base": "https://api.deepseek.com/v1",
                    "timeout": 120, "proxy": "", "custom_headers": {}, "enable": True,
                },
            ],
            "provider": [
                {"id": "llm_manager/main", "enable": True, "provider_source_id": "llm_manager",
                 "model": "main", "modalities": ["text"]},
                {"id": "deepseek/m1", "enable": True, "provider_source_id": "deepseek",
                 "model": "m1", "modalities": ["text"]},
            ],
            "provider_settings": {
                "default_provider_id": "llm_manager/main",
            },
        }
        result = s.import_from_astrbot_config(config)
        self.assertFalse(result["auto_default_set"])
        self.assertEqual(s.data.get("default_instance", ""), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
