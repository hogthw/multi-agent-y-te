"""
Generate Slide-Ready Architecture & Workflow Diagram (Hình 9.1) matching the NTTU Light Slide Palette.
Pure Vietnamese, icon-free, 300 DPI PNG + Vector SVG, proportioned cleanly for the right-hand slide container.
"""

from __future__ import annotations

import os
from PIL import Image, ImageDraw, ImageFont


FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _font(size: int, bold: bool = False):
    path = FONT_BOLD if bold else FONT_REG
    if os.path.exists(path):
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_wh(draw: ImageDraw.ImageDraw, text: str, font, spacing: int = 8):
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return box[2] - box[0], box[3] - box[1]


def _draw_box(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    title: str,
    subtitle: str = "",
    fill: str = "#ffffff",
    border: str = "#132a4a",
    title_color: str = "#0f172a",
    sub_color: str = "#475569",
    title_size: int = 30,
    sub_size: int = 22,
    radius: int = 20,
    border_w: int = 4,
    accent_top: str | None = None,
):
    # Soft drop shadow
    draw.rounded_rectangle([x1 + 5, y1 + 7, x2 + 5, y2 + 7], radius=radius, fill="#e2e8f0")
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=border, width=border_w)
    if accent_top:
        draw.rounded_rectangle([x1 + 2, y1 + 2, x2 - 2, y1 + 14], radius=6, fill=accent_top)

    ft = _font(title_size, bold=True)
    fs = _font(sub_size, bold=False)
    cx = (x1 + x2) / 2
    tw, th = _text_wh(draw, title, ft, spacing=6)

    if subtitle:
        sw, sh = _text_wh(draw, subtitle, fs, spacing=6)
        total_h = th + 12 + sh
        ty = (y1 + y2 - total_h) / 2 + (4 if accent_top else 0)
        draw.multiline_text((cx - tw / 2, ty), title, font=ft, fill=title_color, align="center", spacing=6)
        draw.multiline_text((cx - sw / 2, ty + th + 12), subtitle, font=fs, fill=sub_color, align="center", spacing=6)
    else:
        ty = (y1 + y2 - th) / 2 + (3 if accent_top else 0)
        draw.multiline_text((cx - tw / 2, ty), title, font=ft, fill=title_color, align="center", spacing=6)


def _arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str = "#334155", width: int = 5):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # Arrowhead
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 16
    p1 = (x2, y2)
    p2 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p3 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    draw.polygon([p1, p2, p3], fill=color)


def _pill_label(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, fill: str = "#ffffff", border: str = "#cbd5e1", text_color: str = "#334155", size: int = 20):
    f = _font(size, bold=True)
    tw, th = _text_wh(draw, text, f)
    px, py = 16, 7
    draw.rounded_rectangle([cx - tw / 2 - px, cy - th / 2 - py, cx + tw / 2 + px, cy + th / 2 + py], radius=14, fill=fill, outline=border, width=2)
    draw.text((cx - tw / 2, cy - th / 2 - 1), text, font=f, fill=text_color)


def generate_slide_9_1(output_path: str = "docs/hinh_9_1_luong_4_agent.png") -> str:
    # Proportioned to fit cleanly inside the slide's right-side box (~1600 x 1460)
    W, H = 1600, 1460
    img = Image.new("RGB", (W, H), "#f8fafc")
    draw = ImageDraw.Draw(img)

    # Subtle outer frame matching slide aesthetic
    draw.rounded_rectangle([14, 14, W - 14, H - 14], radius=24, fill="#f8fafc", outline="#cbd5e1", width=3)

    # Step 1: Input FHIR Bundle (Top)
    _draw_box(
        draw, 290, 50, 1310, 175,
        title="Hồ sơ Y tế Chuẩn HL7 FHIR R4 (JSON Bundle)",
        subtitle="Patient  |  Encounter  |  Condition  |  Observation  |  MedicationRequest  |  AllergyIntolerance",
        fill="#132a4a", border="#132a4a", title_color="#ffffff", sub_color="#cbd5e1",
        title_size=31, sub_size=21, radius=18, border_w=3,
    )

    _arrow(draw, 800, 175, 800, 232, color="#132a4a", width=5)

    # Step 2: Two-Tier Memory Vault
    _draw_box(
        draw, 210, 238, 1390, 392,
        title="Kho Bộ Nhớ 2 Tầng (Memory-Augmented · TF-IDF Cosine)",
        subtitle="Bộ nhớ Ngữ nghĩa: Chuẩn FHIR R4, QĐ 130/QĐ-BYT, LOINC/ICD-10, Phác đồ VN\nBộ nhớ Tình huống: Bài học ca cũ & Phòng tránh báo động giả (False Positives)",
        fill="#f0fdfa", border="#0d9488", title_color="#0f766e", sub_color="#134e4a",
        title_size=30, sub_size=21, radius=18, border_w=4,
    )

    # Branching arrows to 3 parallel auditors
    branch_y = 436
    draw.line([(800, 392), (800, branch_y)], fill="#0d9488", width=5)
    draw.line([(310, branch_y), (1290, branch_y)], fill="#0d9488", width=5)
    _arrow(draw, 310, branch_y, 310, 496, color="#0284c7", width=5)
    _arrow(draw, 800, branch_y, 800, 496, color="#7c3aed", width=5)
    _arrow(draw, 1290, branch_y, 1290, 496, color="#e11d48", width=5)
    _pill_label(draw, 800, branch_y, "Truy xuất theo chuyên môn & Rà soát song song (asyncio)", fill="#ffffff", border="#0d9488", text_color="#0f766e", size=20)

    # Step 3: 3 Specialist Auditors in Parallel
    _draw_box(
        draw, 115, 502, 505, 700,
        title="BS. Linh\nKiểm toán Cấu trúc",
        subtitle="Trường bắt buộc FHIR R4\nTham chiếu & QĐ 130",
        fill="#ffffff", border="#0284c7", title_color="#0369a1", sub_color="#334155",
        title_size=27, sub_size=21, radius=18, border_w=4, accent_top="#0284c7",
    )
    _draw_box(
        draw, 605, 502, 995, 700,
        title="BS. Hùng\nMã hóa & Đơn vị",
        subtitle="Đối chiếu ICD-10 / LOINC\nĐơn vị SI (mmol/L)",
        fill="#ffffff", border="#7c3aed", title_color="#6d28d9", sub_color="#334155",
        title_size=27, sub_size=21, radius=18, border_w=4, accent_top="#7c3aed",
    )
    _draw_box(
        draw, 1095, 502, 1485, 700,
        title="BS. Minh\nLâm sàng Chéo",
        subtitle="Thuốc – Dị ứng – Thận\nDengue, Nhi khoa, INR",
        fill="#ffffff", border="#e11d48", title_color="#be123c", sub_color="#334155",
        title_size=27, sub_size=21, radius=18, border_w=4, accent_top="#e11d48",
    )

    # Converging arrows to Meta Agent
    join_y = 755
    draw.line([(310, 700), (310, join_y)], fill="#475569", width=5)
    draw.line([(800, 700), (800, join_y)], fill="#475569", width=5)
    draw.line([(1290, 700), (1290, join_y)], fill="#475569", width=5)
    draw.line([(310, join_y), (1290, join_y)], fill="#475569", width=5)
    _arrow(draw, 800, join_y, 800, 816, color="#0284c7", width=5)

    # Step 4: Meta Agent (BS. Thảo)
    _draw_box(
        draw, 250, 822, 1350, 968,
        title="BS. Thảo — Thẩm định trưởng (Meta Agent)",
        subtitle="Tổng hợp báo cáo 3 chuyên khoa, đối chiếu bằng chứng gốc & gạt bỏ báo động giả",
        fill="#eff6ff", border="#0284c7", title_color="#0369a1", sub_color="#1e3a8a",
        title_size=30, sub_size=22, radius=18, border_w=4,
    )

    _arrow(draw, 800, 968, 800, 1030, color="#d97706", width=5)

    # Step 5: Consensus Review Gate
    _draw_box(
        draw, 330, 1036, 1270, 1175,
        title="Vòng Phản biện Chéo & Bỏ phiếu Đồng thuận (ColaCare)",
        subtitle="Cả 3 kiểm toán viên kiểm tra lại bản tổng hợp: Đồng ý hoặc Phản đối có lý do",
        fill="#fffbeb", border="#d97706", title_color="#b45309", sub_color="#78350f",
        title_size=28, sub_size=21, radius=18, border_w=4,
    )

    # Feedback loop arrow on the left side when < 3/3 consensus
    loop_x = 64
    draw.line([(330, 1105), (loop_x, 1105)], fill="#dc2626", width=5)
    draw.line([(loop_x, 1105), (loop_x, 600)], fill="#dc2626", width=5)
    _arrow(draw, loop_x, 600, 110, 600, color="#dc2626", width=5)

    # Vertical/clean pill label for retry loop (placed cleanly above the orange box corner)
    _pill_label(draw, 165, 1002, "Chưa đủ 3/3\nHọp vòng tiếp", fill="#fef2f2", border="#ef4444", text_color="#b91c1c", size=19)

    # Final arrow down to Output
    _arrow(draw, 800, 1175, 800, 1272, color="#16a34a", width=6)
    _pill_label(draw, 1020, 1222, "Đạt đồng thuận 3/3 (hoặc hết vòng)", fill="#f0fdf4", border="#16a34a", text_color="#15803d", size=20)

    # Step 6: Final Output
    _draw_box(
        draw, 250, 1278, 1350, 1412,
        title="Báo cáo Kiểm toán HL7 FHIR R4 Chuẩn JSON",
        subtitle="Kết luận (Hợp lệ / Danh sách lỗi + Mức độ)  |  Lập luận y khoa  |  Độ trễ & Token",
        fill="#132a4a", border="#16a34a", title_color="#ffffff", sub_color="#bbf7d0",
        title_size=30, sub_size=21, radius=18, border_w=4,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, dpi=(300, 300))
    print(f"Saved slide diagram -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_slide_9_1()
