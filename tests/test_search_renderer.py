"""验证搜索图片筛选内容、全局序号与渲染高度。"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image, ImageDraw

import _astrbot_stub  # noqa: E402,F401  无 astrbot 环境时注入 logger 替身
from astrbot_plugin_llm_manager import renderer
from astrbot_plugin_llm_manager.store import Store


class SearchRendererTest(unittest.TestCase):
    def test_filtered_image_keeps_global_number_and_current_banner(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "backends.json")
            store.add_source("A", "https://a.example/v1", "secret")
            store.add_instance("A", "common", ["gpt-current", "qwen-match", "hidden-model"])
            store.add_instance("A", "hidden", ["other-model"])
            store.add_source("B", "https://a.example/v1", "secret-b")
            store.add_instance("B", "common", ["qwen-match"])
            store.add_source("C", "https://z.example/v1", "secret-c")
            store.add_instance("C", "common", ["unrelated-model"])
            store.set_default("1")
            store.set_instance_enabled("B_common", False)
            snapshot = copy.deepcopy(store.data)
            matches = store.search_models("qwen")
            groups = renderer._group_sources_by_api_base(store, matches)
            self.assertEqual(list(groups), ["https://a.example/v1"])
            self.assertEqual([s["id"] for s in groups["https://a.example/v1"]], ["A", "B"])
            self.assertEqual([i["num"] for i in matches], [2, 5])
            texts = []
            original = ImageDraw.ImageDraw.text

            def record(draw, xy, text, *args, **kwargs):
                texts.append(text)
                return original(draw, xy, text, *args, **kwargs)

            with patch.object(ImageDraw.ImageDraw, "text", record):
                path = Path(renderer.render_catalog_image(store, catalog=matches))
            try:
                with Image.open(path) as image:
                    self.assertEqual(image.size, (renderer.WIDTH, renderer._calc_height(store, matches)))
                    image.verify()
                self.assertEqual(texts.count("qwen-match"), 2)
                self.assertIn("gpt-current", texts)
                self.assertIn("5", texts)
                self.assertIn("停用", texts)
                for hidden in ("hidden-model", "A_hidden", "other-model", "unrelated-model", "https://z.example/v1", "secret"):
                    self.assertNotIn(hidden, texts)
                self.assertEqual(store.data, snapshot)
            finally:
                path.unlink()


if __name__ == "__main__":
    unittest.main()
