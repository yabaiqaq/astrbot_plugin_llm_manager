"""三级模型列表的卡片式图片渲染器（Pillow，零额外依赖）。

渲染结构（从上到下）：
  标题栏：LLM 供应商管理 + 统计
  当前使用中横幅：序号 + 模型名 + 实例id（醒目绿色）
  源卡片（一级）：base_url + 站名
    实例（二级）：实例 ID + 状态标签
      模型（三级）：[序号] 模型名 + 能力标签
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ---------- 布局参数 ----------
WIDTH = 720
PADDING = 16
CARD_RADIUS = 14
TITLE_HEIGHT = 68
CURRENT_BANNER_H = 56
SOURCE_HEADER_H = 46
INSTANCE_HEADER_H = 38
MODEL_ROW_H = 34
CARD_GAP = 14
FOOTER_H = 24

# ---------- 配色 ----------
BG = "#F0F2F5"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E4E7ED"
TITLE_BG_TOP = "#5B7FFF"
TITLE_BG_BOTTOM = "#7B5BFF"
TEXT_PRIMARY = "#1A1B1C"
TEXT_SECONDARY = "#6B7280"
TEXT_MUTED = "#9CA3AF"
ACCENT = "#5B8FF9"
ACCENT_LIGHT = "#E8F1FF"
HIGHLIGHT_BG = "#F0F7FF"
DISABLED_BG = "#F5F5F5"
DISABLED_TEXT = "#9CA3AF"
BADGE_BG = "#5B8FF9"
BADGE_TEXT = "#FFFFFF"
TAG_BG = "#F0F2F5"
TAG_TEXT = "#6B7280"
SUCCESS = "#52C41A"
WARNING = "#FAAD14"

# ---------- 字体 ----------
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:  # noqa: BLE001
                continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Any) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:  # noqa: BLE001
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]


def _truncate(draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int) -> str:
    if _text_width(draw, text, font) <= max_w:
        return text
    ellipsis = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _text_width(draw, text[:mid] + ellipsis, font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ellipsis


def _round_rect(draw: ImageDraw.ImageDraw, xy: tuple, radius: int, **kw: Any) -> None:
    try:
        draw.rounded_rectangle(xy, radius=radius, **kw)
    except Exception:  # noqa: BLE001
        draw.rectangle(xy, **kw)


# ---------- 高度计算 ----------

def _calc_height(store: Any) -> int:
    h = TITLE_HEIGHT + PADDING + CURRENT_BANNER_H + CARD_GAP
    for src in store.sources():
        h += SOURCE_HEADER_H
        for inst in src.get("instances", []):
            h += INSTANCE_HEADER_H
            models = inst.get("models", []) or []
            h += MODEL_ROW_H * max(len(models), 1)
        h += CARD_GAP
    h += FOOTER_H
    return h


# ---------- 主渲染 ----------

def render_catalog_image(store: Any, umo: str | None = None) -> str:
    """渲染三级目录为 PNG 图片，返回临时文件路径。"""
    catalog = store.build_catalog()
    total_models = len(catalog)
    total_sources = len(store.sources())
    default_inst = store.data.get("default_instance", "")
    default_model = store.data.get("default_model", "")
    override_id = store.get_conversation_override(umo) if umo else None

    # 解析当前真正生效的模型（优先级：请求指定 → 会话override → 全局默认 → 兜底）
    current_routed = store.resolve(umo=umo)
    current_instance_id = current_routed["instance"]["id"] if current_routed else ""
    current_model_name = current_routed["model_name"] if current_routed else ""
    # 找到当前模型的全局序号
    current_num = None
    for item in catalog:
        if item["instance_id"] == current_instance_id and item["model"] == current_model_name:
            current_num = item["num"]
            break

    height = _calc_height(store)
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)

    # 字体
    f_title = _load_font(22)
    f_subtitle = _load_font(13)
    f_source = _load_font(15)
    f_source_small = _load_font(12)
    f_instance = _load_font(14)
    f_model = _load_font(13)
    f_badge = _load_font(12)
    f_tag = _load_font(10)

    y = 0

    # ===== 标题栏 =====
    draw.rectangle([0, 0, WIDTH, TITLE_HEIGHT], fill=TITLE_BG_TOP)
    # 渐变效果（用多条横线模拟）
    for i in range(TITLE_HEIGHT):
        r = int(0x5B + (0x7B - 0x5B) * i / TITLE_HEIGHT)
        g = int(0x7F + (0x5B - 0x7F) * i / TITLE_HEIGHT)
        b = int(0xFF + (0xFF - 0xFF) * i / TITLE_HEIGHT)
        draw.line([(0, i), (WIDTH, i)], fill=(r, g, b))
    draw.text((PADDING + 4, 14), "LLM 供应商管理", font=f_title, fill="#FFFFFF")
    stat = f"{total_sources} 个源 · {total_models} 个模型"
    sw = _text_width(draw, stat, f_subtitle)
    draw.text((WIDTH - PADDING - sw - 4, 24), stat, font=f_subtitle, fill="#E0E7FF")
    y = TITLE_HEIGHT + PADDING

    # ===== 当前使用中横幅（醒目绿色）=====
    banner_top = y
    banner_bottom = y + CURRENT_BANNER_H
    # 浅绿色背景 + 绿色边框
    _round_rect(draw, (PADDING, banner_top, WIDTH - PADDING, banner_bottom),
                10, fill="#E8F7E8", outline="#52C41A", width=2)
    # 左侧绿色圆点 + "当前使用中"
    draw.ellipse((PADDING + 18, banner_top + 20, PADDING + 30, banner_top + 32),
                 fill="#52C41A")
    draw.text((PADDING + 38, banner_top + 16), "当前使用中", font=f_source,
              fill="#237804")
    # 右侧：序号徽章 + 模型名 + 实例id
    if current_num is not None and current_model_name:
        # 序号徽章（绿色）
        num_text = str(current_num)
        nw = max(30, _text_width(draw, num_text, f_badge) + 14)
        _round_rect(draw, (WIDTH - PADDING - 200 - nw, banner_top + 14,
                           WIDTH - PADDING - 200, banner_top + CURRENT_BANNER_H - 14),
                    6, fill="#52C41A")
        ntx = WIDTH - PADDING - 200 - nw + (nw - _text_width(draw, num_text, f_badge)) // 2
        draw.text((ntx, banner_top + 17), num_text, font=f_badge, fill="#FFFFFF")
        # 模型名
        model_text = _truncate(draw, current_model_name, f_source, 180)
        draw.text((WIDTH - PADDING - 190, banner_top + 16), model_text,
                  font=f_source, fill="#1A1B1C")
        # 实例id（小字）
        inst_text = _truncate(draw, current_instance_id, f_tag, 190)
        draw.text((WIDTH - PADDING - 190, banner_top + 36), inst_text,
                  font=f_tag, fill="#6B7280")
    else:
        draw.text((WIDTH - PADDING - 200, banner_top + 18), "（未配置模型）",
                  font=f_source, fill="#9CA3AF")
    y = banner_bottom + CARD_GAP

    # ===== 空态 =====
    if total_sources == 0:
        card_h = 100
        _round_rect(draw, (PADDING, y, WIDTH - PADDING, y + card_h), CARD_RADIUS,
                    fill=CARD_BG, outline=CARD_BORDER)
        msg = "尚未配置任何后端"
        mw = _text_width(draw, msg, f_instance)
        draw.text(((WIDTH - mw) // 2, y + 30), msg, font=f_instance, fill=TEXT_SECONDARY)
        hint = "用 /llm import 从系统配置导入，或 /llm add 手动添加"
        hw = _text_width(draw, hint, f_tag)
        draw.text(((WIDTH - hw) // 2, y + 58), hint, font=f_tag, fill=TEXT_MUTED)
        return _save(img)

    # ===== 源卡片 =====
    for src in store.sources():
        src_enabled = src.get("enabled", True)
        insts = src.get("instances", [])
        card_h = SOURCE_HEADER_H
        for inst in insts:
            card_h += INSTANCE_HEADER_H
            models = inst.get("models", []) or []
            card_h += MODEL_ROW_H * max(len(models), 1)

        card_top = y
        card_bottom = y + card_h
        _round_rect(draw, (PADDING, card_top, WIDTH - PADDING, card_bottom),
                    CARD_RADIUS, fill=CARD_BG, outline=CARD_BORDER)

        # --- 源标题（一级）---
        header_y = card_top
        api_base = str(src.get("api_base", "?"))
        site = str(src.get("site", ""))
        base_text = _truncate(draw, api_base, f_source, WIDTH - PADDING * 2 - 120)
        draw.text((PADDING + 14, header_y + 13), base_text, font=f_source,
                  fill=TEXT_PRIMARY if src_enabled else DISABLED_TEXT)
        # 站名 + 类型
        site_tag = f"{site} · {src.get('type', '?')}"
        site_tag = _truncate(draw, site_tag, f_source_small, 180)
        stw = _text_width(draw, site_tag, f_source_small)
        draw.text((WIDTH - PADDING - stw - 14, header_y + 16), site_tag,
                  font=f_source_small, fill=TEXT_MUTED if src_enabled else DISABLED_TEXT)
        # 分隔线
        draw.line([(PADDING + 10, header_y + SOURCE_HEADER_H - 1),
                   (WIDTH - PADDING - 10, header_y + SOURCE_HEADER_H - 1)],
                  fill="#F0F2F5")

        iy = header_y + SOURCE_HEADER_H

        # --- 实例（二级）---
        for inst in insts:
            inst_id = str(inst.get("id", "?"))
            inst_enabled = inst.get("enabled", True) and src_enabled
            is_default = (inst_id == default_inst and default_model == "")
            is_override = (inst_id == override_id)
            row_bg = HIGHLIGHT_BG if (is_default or is_override) else None

            if row_bg:
                draw.rectangle([PADDING + 1, iy, WIDTH - PADDING - 1,
                                iy + INSTANCE_HEADER_H], fill=row_bg)

            draw.text((PADDING + 28, iy + 9), "└─", font=f_instance,
                      fill=TEXT_MUTED if inst_enabled else DISABLED_TEXT)
            inst_text = _truncate(draw, inst_id, f_instance, WIDTH - PADDING * 2 - 200)
            draw.text((PADDING + 52, iy + 9), inst_text, font=f_instance,
                      fill=TEXT_PRIMARY if inst_enabled else DISABLED_TEXT)

            # 状态标签
            tag_x = PADDING + 52 + _text_width(draw, inst_text, f_instance) + 8
            if is_override:
                _draw_tag(draw, tag_x, iy + 10, "本会话", f_tag, ACCENT, "#FFFFFF")
                tag_x += 60
            elif is_default:
                _draw_tag(draw, tag_x, iy + 10, "默认", f_tag, SUCCESS, "#FFFFFF")
                tag_x += 48
            if not inst_enabled:
                _draw_tag(draw, tag_x, iy + 10, "停用", f_tag, "#D9D9D9", "#999999")

            iy += INSTANCE_HEADER_H

            # --- 模型（三级）---
            models = inst.get("models", []) or []
            if not models:
                draw.text((PADDING + 64, iy + 8), "（暂无模型）", font=f_model,
                          fill=TEXT_MUTED)
                iy += MODEL_ROW_H
                continue

            for item in catalog:
                if item["instance_id"] != inst_id:
                    continue
                model_name = item["model"]
                disabled = item["disabled"]
                is_cur = (not disabled and inst_id == current_instance_id
                          and model_name == current_model_name)
                mrow_bg = HIGHLIGHT_BG if is_cur else None
                if mrow_bg:
                    draw.rectangle([PADDING + 1, iy, WIDTH - PADDING - 1,
                                    iy + MODEL_ROW_H], fill=mrow_bg)

                # 序号徽章
                badge_text = str(item["num"])
                bw = max(26, _text_width(draw, badge_text, f_badge) + 12)
                _round_rect(draw, (PADDING + 64, iy + 6,
                                   PADDING + 64 + bw, iy + MODEL_ROW_H - 6),
                            6, fill=BADGE_BG if not disabled else "#D9D9D9")
                btx = PADDING + 64 + (bw - _text_width(draw, badge_text, f_badge)) // 2
                draw.text((btx, iy + 8), badge_text, font=f_badge, fill=BADGE_TEXT)

                # 模型名
                mx = PADDING + 64 + bw + 10
                model_text = _truncate(draw, model_name, f_model,
                                        WIDTH - mx - PADDING - 120)
                draw.text((mx, iy + 8), model_text, font=f_model,
                          fill=TEXT_PRIMARY if not disabled else DISABLED_TEXT)

                # 能力标签
                mods = item.get("modalities", [])
                tag_x2 = mx + _text_width(draw, model_text, f_model) + 8
                if "tool_use" in mods:
                    _draw_tag(draw, tag_x2, iy + 9, "工具", f_tag, TAG_BG, TAG_TEXT)
                    tag_x2 += 42
                if "image" in mods:
                    _draw_tag(draw, tag_x2, iy + 9, "视觉", f_tag, TAG_BG, TAG_TEXT)
                    tag_x2 += 42
                if "audio" in mods:
                    _draw_tag(draw, tag_x2, iy + 9, "语音", f_tag, TAG_BG, TAG_TEXT)

                # 当前使用中标记（绿色圆点 + 文字，醒目）
                if is_cur:
                    _draw_tag(draw, WIDTH - PADDING - 78, iy + 8, "● 使用中",
                              f_tag, "#52C41A", "#FFFFFF")
                elif disabled:
                    dis_text = "停用"
                    dw = _text_width(draw, dis_text, f_tag)
                    draw.text((WIDTH - PADDING - dw - 14, iy + 10), dis_text,
                              font=f_tag, fill=DISABLED_TEXT)

                iy += MODEL_ROW_H

        y = card_bottom + CARD_GAP

    return _save(img)


def _draw_tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
              font: Any, bg: str, fg: str) -> None:
    tw = _text_width(draw, text, font)
    pad_x, pad_y = 6, 3
    _round_rect(draw, (x, y, x + tw + pad_x * 2, y + font.size + pad_y * 2 + 2),
                4, fill=bg)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=fg)


def _save(img: Image.Image) -> str:
    fd, path = tempfile.mkstemp(suffix=".png", prefix="llm_manager_")
    os.close(fd)
    img.save(path, "PNG", optimize=True)
    return path
