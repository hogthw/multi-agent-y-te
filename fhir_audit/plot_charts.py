"""
Publication-Ready Scientific Chart Generator (300 DPI PNG + Vector SVG, Light Slide Theme).

Reads logs/fhir_audit/results/benchmark_summary.json and generates 3 clean,
pure-Vietnamese, icon-free charts ready to embed into PPNCKH thesis reports & slides:
  1. chart1_f1_fpr_comparison.png / .svg  : F1-score (with Bootstrap +/- Std error bars) & FPR (%)
  2. chart2_category_recall.png   / .svg  : Per-category Recall (%) across Structural / Terminology / Clinical Semantic
  3. chart3_efficiency_tradeoff.png / .svg: Effectiveness vs. Efficiency Trade-off (F1 vs. Latency & Token bubble size)

Uses Pillow (PIL) with macOS/system TrueType Vietnamese fonts — zero matplotlib dependency required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SUMMARY_PATH = os.path.join("logs", "fhir_audit", "results", "benchmark_summary.json")
DEFAULT_CHARTS_DIR = os.path.join("logs", "fhir_audit", "results", "charts")

METHOD_SHORT_VI = {
    "rule_based": "Luật FHIR\n(Đối chứng 1)",
    "single_llm": "Single-LLM\n(Zero-Shot)",
    "single_llm_memory": "Single-LLM\n+ Bộ nhớ",
    "multi_agent_no_mem": "Multi-Agent\n(Không bộ nhớ)",
    "multi_agent_sem": "Multi-Agent\n+ Ngữ nghĩa",
    "multi_agent_memory": "Multi-Agent\n+ Bộ nhớ 2 tầng",
}

METHOD_INLINE_VI = {
    "rule_based": "Luật FHIR (Rule)",
    "single_llm": "Single-LLM",
    "single_llm_memory": "Single-LLM + Bộ nhớ",
    "multi_agent_no_mem": "Multi-Agent (No Mem)",
    "multi_agent_sem": "Multi-Agent + Ngữ nghĩa",
    "multi_agent_memory": "Multi-Agent + Bộ nhớ 2 tầng",
}

FONT_REGULAR_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

FONT_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_REGULAR_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> Tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered_text(draw: ImageDraw.ImageDraw, cx: float, y: float, text: str, font: Any, fill: str) -> None:
    w, _ = _text_size(draw, text, font)
    draw.multiline_text((cx - w / 2, y), text, font=font, fill=fill, align="center", spacing=6)


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# =============================================================================
# CHART 1: F1-SCORE (WITH BOOTSTRAP ERROR BARS) & FALSE POSITIVE RATE (FPR)
# =============================================================================
def draw_chart1_f1_fpr(summaries: List[Dict[str, Any]], model_key: str, out_dir: str) -> Tuple[str, str]:
    W, H = 2200, 1280
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = _load_font(44, bold=True)
    f_sub = _load_font(28, bold=False)
    f_axis = _load_font(26, bold=True)
    f_tick = _load_font(24, bold=False)
    f_val = _load_font(23, bold=True)
    f_leg = _load_font(25, bold=False)

    title = "Biểu đồ 1. So sánh F1-score (Bootstrap 1.000 lần) và Tỷ lệ Báo động giả (FPR)"
    subtitle = f"Tập chuẩn HL7 FHIR R4 (Quốc tế & Việt Nam) | Mô hình nền: {model_key or 'N/A'} | Đơn vị: %"
    _draw_centered_text(draw, W / 2, 42, title, f_title, "#0f172a")
    _draw_centered_text(draw, W / 2, 102, subtitle, f_sub, "#475569")

    # Legend
    leg_y = 162
    legend_items = [
        ("#0284c7", "F1-score — Các phương pháp đối chứng (%)"),
        ("#0d9488", "F1-score — Hệ thống đề xuất (+ Bộ nhớ 2 tầng)"),
        ("#ef4444", "Tỷ lệ báo động giả FPR (%) — càng thấp càng tốt"),
    ]
    lx = 240
    for col, lbl in legend_items:
        draw.rectangle([lx, leg_y + 4, lx + 30, leg_y + 28], fill=col, outline="#334155", width=1)
        draw.text((lx + 40, leg_y), lbl, font=f_leg, fill="#1e293b")
        w_lbl, _ = _text_size(draw, lbl, f_leg)
        lx += w_lbl + 76

    # Plot area
    left, top, right, bottom = 180, 235, W - 90, H - 210
    plot_w = right - left
    plot_h = bottom - top

    # Horizontal grid lines (0% .. 100%)
    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        draw.line([(left, y), (right, y)], fill="#e2e8f0" if pct > 0 else "#334155", width=2 if pct > 0 else 3)
        tw, th = _text_size(draw, f"{pct}%", f_tick)
        draw.text((left - tw - 18, y - th / 2), f"{pct}%", font=f_tick, fill="#334155")

    draw.line([(left, top), (left, bottom)], fill="#334155", width=3)
    draw.text((38, top - 45), "Tỷ lệ (%)", font=f_axis, fill="#0f172a")

    n = max(1, len(summaries))
    slot_w = plot_w / n
    bar_w = min(105, slot_w * 0.28)
    gap = max(12, bar_w * 0.16)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="82" text-anchor="middle" font-size="40" font-weight="700" fill="#0f172a">{_xml_escape(title)}</text>',
        f'<text x="{W/2}" y="128" text-anchor="middle" font-size="25" fill="#475569">{_xml_escape(subtitle)}</text>',
    ]
    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        svg_parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{"#e2e8f0" if pct > 0 else "#334155"}" stroke-width="{2 if pct > 0 else 3}"/>')
        svg_parts.append(f'<text x="{left-16}" y="{y+8}" text-anchor="end" font-size="23" fill="#334155">{pct}%</text>')
    svg_parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155" stroke-width="3"/>')

    for i, s in enumerate(summaries):
        cx = left + (i + 0.5) * slot_w
        f1 = float(s.get("f1", 0.0))
        std = float(s.get("f1_bootstrap_std", 0.0))
        fpr = float(s.get("fpr", 0.0))
        is_proposed = s.get("method") == "multi_agent_memory"
        f1_col = "#0d9488" if is_proposed else "#0284c7"

        # F1 bar
        x1_l = cx - gap / 2 - bar_w
        x1_r = cx - gap / 2
        y1 = bottom - (min(100.0, f1) / 100.0) * plot_h
        draw.rectangle([x1_l, y1, x1_r, bottom], fill=f1_col, outline="#0f172a", width=2)
        svg_parts.append(f'<rect x="{x1_l:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{bottom-y1:.1f}" fill="{f1_col}" stroke="#0f172a" stroke-width="2"/>')

        # Bootstrap +/- Std error bar
        if std > 0:
            ex = (x1_l + x1_r) / 2
            ey_hi = bottom - (min(100.0, f1 + std) / 100.0) * plot_h
            ey_lo = bottom - (max(0.0, f1 - std) / 100.0) * plot_h
            draw.line([(ex, ey_hi), (ex, ey_lo)], fill="#0f172a", width=3)
            draw.line([(ex - 12, ey_hi), (ex + 12, ey_hi)], fill="#0f172a", width=3)
            draw.line([(ex - 12, ey_lo), (ex + 12, ey_lo)], fill="#0f172a", width=3)
            svg_parts.append(f'<line x1="{ex:.1f}" y1="{ey_hi:.1f}" x2="{ex:.1f}" y2="{ey_lo:.1f}" stroke="#0f172a" stroke-width="3"/>')
            svg_parts.append(f'<line x1="{ex-12:.1f}" y1="{ey_hi:.1f}" x2="{ex+12:.1f}" y2="{ey_hi:.1f}" stroke="#0f172a" stroke-width="3"/>')
            svg_parts.append(f'<line x1="{ex-12:.1f}" y1="{ey_lo:.1f}" x2="{ex+12:.1f}" y2="{ey_lo:.1f}" stroke="#0f172a" stroke-width="3"/>')
            val_y = min(y1, ey_hi) - 36
        else:
            val_y = y1 - 36

        _draw_centered_text(draw, (x1_l + x1_r) / 2, max(top - 30, val_y), f"{f1:.1f}%", f_val, "#0f172a")
        svg_parts.append(f'<text x="{(x1_l+x1_r)/2:.1f}" y="{max(top-10, val_y+20):.1f}" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{f1:.1f}%</text>')

        # FPR bar
        x2_l = cx + gap / 2
        x2_r = cx + gap / 2 + bar_w
        y2 = bottom - (min(100.0, fpr) / 100.0) * plot_h
        if fpr > 0:
            draw.rectangle([x2_l, y2, x2_r, bottom], fill="#ef4444", outline="#7f1d1d", width=2)
            svg_parts.append(f'<rect x="{x2_l:.1f}" y="{y2:.1f}" width="{bar_w:.1f}" height="{bottom-y2:.1f}" fill="#ef4444" stroke="#7f1d1d" stroke-width="2"/>')
        _draw_centered_text(draw, (x2_l + x2_r) / 2, y2 - 36, f"{fpr:.1f}%", f_val, "#b91c1c")
        svg_parts.append(f'<text x="{(x2_l+x2_r)/2:.1f}" y="{y2-14:.1f}" text-anchor="middle" font-size="22" font-weight="700" fill="#b91c1c">{fpr:.1f}%</text>')

        # X-axis method label
        lbl = METHOD_SHORT_VI.get(s["method"], s["method"])
        _draw_centered_text(draw, cx, bottom + 22, lbl, f_axis if is_proposed else f_tick, "#0f766e" if is_proposed else "#1e293b")
        for li, line in enumerate(lbl.split("\n")):
            svg_parts.append(f'<text x="{cx:.1f}" y="{bottom+46+li*30:.1f}" text-anchor="middle" font-size="23" font-weight="{"700" if is_proposed else "400"}" fill="{"#0f766e" if is_proposed else "#1e293b"}">{_xml_escape(line)}</text>')

    svg_parts.append("</svg>")
    png_path = os.path.join(out_dir, "chart1_f1_fpr_comparison.png")
    svg_path = os.path.join(out_dir, "chart1_f1_fpr_comparison.svg")
    img.save(png_path, dpi=(300, 300))
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    return png_path, svg_path


# =============================================================================
# CHART 2: PER-CATEGORY RECALL (STRUCTURAL vs TERMINOLOGY vs CLINICAL SEMANTIC)
# =============================================================================
def draw_chart2_category_recall(summaries: List[Dict[str, Any]], model_key: str, out_dir: str) -> Tuple[str, str]:
    W, H = 2200, 1280
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = _load_font(44, bold=True)
    f_sub = _load_font(28, bold=False)
    f_axis = _load_font(26, bold=True)
    f_tick = _load_font(24, bold=False)
    f_val = _load_font(21, bold=True)
    f_leg = _load_font(25, bold=False)

    title = "Biểu đồ 2. Độ phủ phát hiện lỗi (Recall %) theo 3 nhóm lỗi HL7 FHIR R4"
    subtitle = f"Phân tách Cấu trúc vs. Mã hóa & Đơn vị vs. Ngữ nghĩa Lâm sàng Chéo | Mô hình: {model_key or 'N/A'}"
    _draw_centered_text(draw, W / 2, 42, title, f_title, "#0f172a")
    _draw_centered_text(draw, W / 2, 102, subtitle, f_sub, "#475569")

    cats = [
        ("STRUCTURAL", "Lỗi Cấu trúc & QĐ 130 (Structural)", "#0284c7"),
        ("TERMINOLOGY", "Lỗi Mã hóa & Đơn vị SI (Terminology)", "#7c3aed"),
        ("CLINICAL_SEMANTIC", "Lỗi Ngữ nghĩa Lâm sàng Chéo (Clinical Semantic)", "#e11d48"),
    ]

    lx = 210
    leg_y = 162
    for _, lbl, col in cats:
        draw.rectangle([lx, leg_y + 4, lx + 30, leg_y + 28], fill=col, outline="#1e293b", width=1)
        draw.text((lx + 40, leg_y), lbl, font=f_leg, fill="#1e293b")
        w_lbl, _ = _text_size(draw, lbl, f_leg)
        lx += w_lbl + 60

    left, top, right, bottom = 180, 235, W - 90, H - 210
    plot_w = right - left
    plot_h = bottom - top

    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        draw.line([(left, y), (right, y)], fill="#e2e8f0" if pct > 0 else "#334155", width=2 if pct > 0 else 3)
        tw, th = _text_size(draw, f"{pct}%", f_tick)
        draw.text((left - tw - 18, y - th / 2), f"{pct}%", font=f_tick, fill="#334155")
    draw.line([(left, top), (left, bottom)], fill="#334155", width=3)
    draw.text((28, top - 45), "Recall (%)", font=f_axis, fill="#0f172a")

    n = max(1, len(summaries))
    slot_w = plot_w / n
    bar_w = min(78, slot_w * 0.22)
    gap = max(8, bar_w * 0.14)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="82" text-anchor="middle" font-size="40" font-weight="700" fill="#0f172a">{_xml_escape(title)}</text>',
        f'<text x="{W/2}" y="128" text-anchor="middle" font-size="25" fill="#475569">{_xml_escape(subtitle)}</text>',
    ]
    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        svg_parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{"#e2e8f0" if pct > 0 else "#334155"}" stroke-width="{2 if pct > 0 else 3}"/>')
        svg_parts.append(f'<text x="{left-16}" y="{y+8}" text-anchor="end" font-size="23" fill="#334155">{pct}%</text>')
    svg_parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155" stroke-width="3"/>')

    for i, s in enumerate(summaries):
        cx = left + (i + 0.5) * slot_w
        pc = s.get("per_category", {})
        is_proposed = s.get("method") == "multi_agent_memory"
        group_w = 3 * bar_w + 2 * gap
        start_x = cx - group_w / 2

        for ci, (ckey, _, col) in enumerate(cats):
            val = float(pc.get(ckey, {}).get("category_recall", 0.0))
            bx_l = start_x + ci * (bar_w + gap)
            bx_r = bx_l + bar_w
            by = bottom - (min(100.0, val) / 100.0) * plot_h
            if val > 0:
                draw.rectangle([bx_l, by, bx_r, bottom], fill=col, outline="#0f172a", width=2)
                svg_parts.append(f'<rect x="{bx_l:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bottom-by:.1f}" fill="{col}" stroke="#0f172a" stroke-width="2"/>')
            _draw_centered_text(draw, (bx_l + bx_r) / 2, by - 32, f"{val:.0f}%", f_val, "#0f172a" if val > 0 else "#94a3b8")
            svg_parts.append(f'<text x="{(bx_l+bx_r)/2:.1f}" y="{by-12:.1f}" text-anchor="middle" font-size="20" font-weight="700" fill="{"#0f172a" if val > 0 else "#94a3b8"}">{val:.0f}%</text>')

        lbl = METHOD_SHORT_VI.get(s["method"], s["method"])
        _draw_centered_text(draw, cx, bottom + 22, lbl, f_axis if is_proposed else f_tick, "#0f766e" if is_proposed else "#1e293b")
        for li, line in enumerate(lbl.split("\n")):
            svg_parts.append(f'<text x="{cx:.1f}" y="{bottom+46+li*30:.1f}" text-anchor="middle" font-size="23" font-weight="{"700" if is_proposed else "400"}" fill="{"#0f766e" if is_proposed else "#1e293b"}">{_xml_escape(line)}</text>')

    svg_parts.append("</svg>")
    png_path = os.path.join(out_dir, "chart2_category_recall.png")
    svg_path = os.path.join(out_dir, "chart2_category_recall.svg")
    img.save(png_path, dpi=(300, 300))
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    return png_path, svg_path


# =============================================================================
# CHART 3: EFFICIENCY TRADE-OFF (F1 vs LATENCY & TOKEN BUBBLE SIZE)
# =============================================================================
def draw_chart3_tradeoff(summaries: List[Dict[str, Any]], model_key: str, out_dir: str) -> Tuple[str, str]:
    W, H = 2200, 1280
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = _load_font(44, bold=True)
    f_sub = _load_font(28, bold=False)
    f_axis = _load_font(26, bold=True)
    f_tick = _load_font(24, bold=False)
    f_lbl = _load_font(23, bold=True)
    f_small = _load_font(20, bold=False)

    title = "Biểu đồ 3. Phân tích đánh đổi hiệu năng: F1-score so với Độ trễ và Chi phí Token"
    subtitle = f"Kích thước vòng tròn tỷ lệ với số Token trung bình/hồ sơ | Góc trên bên trái là vùng tối ưu | Mô hình: {model_key or 'N/A'}"
    _draw_centered_text(draw, W / 2, 42, title, f_title, "#0f172a")
    _draw_centered_text(draw, W / 2, 102, subtitle, f_sub, "#475569")

    left, top, right, bottom = 190, 210, W - 150, H - 170
    plot_w = right - left
    plot_h = bottom - top

    max_lat = max([float(s.get("mean_latency_s", 0.0)) for s in summaries] + [2.0]) * 1.25
    max_lat = max(5.0, math.ceil(max_lat / 2.0) * 2.0)
    max_tok = max([int(s.get("mean_tokens", 0)) for s in summaries] + [1000])

    # Subtle highlight zone in top-left (high F1, low latency)
    draw.rectangle([left, top, left + plot_w * 0.45, top + plot_h * 0.3], fill="#f0fdfa")
    draw.text((left + 20, top + 16), "Vùng tối ưu (F1 cao, độ trễ thấp)", font=f_small, fill="#0f766e")

    # Grid Y (0..100%)
    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        draw.line([(left, y), (right, y)], fill="#e2e8f0" if pct > 0 else "#334155", width=2 if pct > 0 else 3)
        tw, th = _text_size(draw, f"{pct}%", f_tick)
        draw.text((left - tw - 18, y - th / 2), f"{pct}%", font=f_tick, fill="#334155")

    # Grid X (0..max_lat)
    steps_x = 5
    for i in range(steps_x + 1):
        val = (max_lat / steps_x) * i
        x = left + (i / steps_x) * plot_w
        draw.line([(x, top), (x, bottom)], fill="#e2e8f0" if i > 0 else "#334155", width=2 if i > 0 else 3)
        _draw_centered_text(draw, x, bottom + 18, f"{val:.1f}s", f_tick, "#334155")

    draw.text((34, top - 45), "F1-score (%)", font=f_axis, fill="#0f172a")
    _draw_centered_text(draw, (left + right) / 2, bottom + 72, "Độ trễ trung bình mỗi hồ sơ FHIR (giây/bundle)", f_axis, "#0f172a")

    palette = {
        "rule_based": "#64748b",
        "single_llm": "#f59e0b",
        "single_llm_memory": "#0284c7",
        "multi_agent_no_mem": "#7c3aed",
        "multi_agent_sem": "#e11d48",
        "multi_agent_memory": "#0d9488",
    }

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="82" text-anchor="middle" font-size="40" font-weight="700" fill="#0f172a">{_xml_escape(title)}</text>',
        f'<text x="{W/2}" y="128" text-anchor="middle" font-size="25" fill="#475569">{_xml_escape(subtitle)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w*0.45:.1f}" height="{plot_h*0.3:.1f}" fill="#f0fdfa"/>',
    ]
    for pct in range(0, 101, 20):
        y = bottom - (pct / 100.0) * plot_h
        svg_parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{"#e2e8f0" if pct > 0 else "#334155"}" stroke-width="{2 if pct > 0 else 3}"/>')
        svg_parts.append(f'<text x="{left-16}" y="{y+8}" text-anchor="end" font-size="23" fill="#334155">{pct}%</text>')
    for i in range(steps_x + 1):
        val = (max_lat / steps_x) * i
        x = left + (i / steps_x) * plot_w
        svg_parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="{"#e2e8f0" if i > 0 else "#334155"}" stroke-width="{2 if i > 0 else 3}"/>')
        svg_parts.append(f'<text x="{x:.1f}" y="{bottom+40}" text-anchor="middle" font-size="23" fill="#334155">{val:.1f}s</text>')

    for idx, s in enumerate(summaries):
        lat = float(s.get("mean_latency_s", 0.0))
        f1 = float(s.get("f1", 0.0))
        tok = int(s.get("mean_tokens", 0))
        col = palette.get(s["method"], "#0284c7")

        cx = left + (min(max_lat, lat) / max_lat) * plot_w
        cy = bottom - (min(100.0, f1) / 100.0) * plot_h
        radius = 24 + int(42 * math.sqrt(tok / max(1, max_tok)))

        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=col, outline="#0f172a", width=3)
        svg_parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{col}" fill-opacity="0.88" stroke="#0f172a" stroke-width="3"/>')

        name = METHOD_INLINE_VI.get(s["method"], s["method"])
        detail = f"F1={f1:.1f}% | {lat:.1f}s | {tok:,} tok"
        # Offset label neatly so nearby points don't collide
        dy_off = -radius - 64 if idx % 2 == 0 else radius + 12
        if cy + dy_off < top + 10:
            dy_off = radius + 12
        if cy + dy_off > bottom - 55:
            dy_off = -radius - 64

        lx = max(left + 120, min(right - 140, cx))
        _draw_centered_text(draw, lx, cy + dy_off, name, f_lbl, "#0f172a")
        _draw_centered_text(draw, lx, cy + dy_off + 28, detail, f_small, "#475569")
        svg_parts.append(f'<text x="{lx:.1f}" y="{cy+dy_off+20:.1f}" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{_xml_escape(name)}</text>')
        svg_parts.append(f'<text x="{lx:.1f}" y="{cy+dy_off+46:.1f}" text-anchor="middle" font-size="19" fill="#475569">{_xml_escape(detail)}</text>')

    svg_parts.append("</svg>")
    png_path = os.path.join(out_dir, "chart3_efficiency_tradeoff.png")
    svg_path = os.path.join(out_dir, "chart3_efficiency_tradeoff.svg")
    img.save(png_path, dpi=(300, 300))
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    return png_path, svg_path


def generate_all_charts(
    summary_path: str = DEFAULT_SUMMARY_PATH,
    output_dir: str = DEFAULT_CHARTS_DIR,
) -> List[str]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Không tìm thấy tệp kết quả benchmark: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = data.get("summaries", [])
    model_key = data.get("model_key", "N/A")
    if not summaries:
        raise ValueError("Chưa có dữ liệu summaries trong benchmark_summary.json")

    os.makedirs(output_dir, exist_ok=True)
    generated: List[str] = []
    for fn in (draw_chart1_f1_fpr, draw_chart2_category_recall, draw_chart3_tradeoff):
        png_p, svg_p = fn(summaries, model_key, output_dir)
        generated.extend([png_p, svg_p])
    print("Generated publication-ready charts:")
    for p in generated:
        print(f"  - {p}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 300-DPI PNG + SVG charts from FHIR benchmark results.")
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CHARTS_DIR)
    args = parser.parse_args()
    generate_all_charts(args.summary, args.output_dir)
