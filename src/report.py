"""One-page PDF report of a clinician-approved assessment (Pillow only, no UI code).

render_report() lays out an A4 page as an image; save_report_pdf() writes it
as a PDF. Light colours, so it prints well. Only call these after the
clinician has approved the findings (HITL): the report states who approved
them and when.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from analysis import LOWER_ARCH, UPPER_ARCH
from inference import IMPACTED_MODEL, NUMBERING_MODEL
from overlay import draw_analysis_overlay, font
from theme import CROWDING_BOX, DANGER, SPACING_BOX, TOOTH_BOX

DPI = 150
PAGE_W, PAGE_H = 1240, 1754  # A4 at 150 dpi
MARGIN = 70
CONTENT_W = PAGE_W - 2 * MARGIN
MAX_NOTES_CHARS = 1500

INK = "#1A1A1A"
MUTED = "#5F5F5F"
RULE = "#C9C9C9"
PANEL = "#F3F4F6"
FAINT = "#BDBDBD"  # missing third molar on the chart

DISCLAIMER = (
    "Decision support only. Findings are AI suggestions reviewed and approved by the clinician "
    "named above; they are not a diagnosis. Spacing and overcrowding are derived from tooth "
    "numbering (missing or repeated FDI numbers), not from measurements."
)


@dataclass
class ReportData:
    image: Image.Image  # the X-ray as analyzed
    analysis: object  # analysis.AnalysisResult
    clinician: str
    approved_at: datetime
    case_reference: str = ""
    notes: str = ""
    app_version: str = ""
    generated_at: datetime = field(default_factory=datetime.now)


def save_report_pdf(data, path):
    """Render the report and write it as a one-page PDF (atomically)."""
    path = Path(path)
    page = render_report(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    page.save(tmp, "PDF", resolution=float(DPI), title="OrthoScan AI assessment report")
    os.replace(tmp, path)
    return path


def render_report(data):
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN

    # --- Header ------------------------------------------------------------------
    draw.text((MARGIN, y), "OrthoScan AI — Orthodontic Assessment Report", font=font(34), fill=INK)
    y += 50
    draw.text((MARGIN, y), "AI-assisted assessment — clinician verified", font=font(18, bold=False), fill=MUTED)
    y += 40
    case = data.case_reference.strip() or "—"
    meta = [
        ("Case reference", case),
        ("Report date", f"{data.generated_at:%d %b %Y, %H:%M}"),
        ("X-ray", f"{data.image.width} × {data.image.height} px"),
        ("Analysis time", f"{data.analysis.timings_ms.get('total', 0) / 1000:.1f} s"),
    ]
    col_w = CONTENT_W // len(meta)
    for i, (label, value) in enumerate(meta):
        x = MARGIN + i * col_w
        draw.text((x, y), label, font=font(14, bold=False), fill=MUTED)
        draw.text((x, y + 20), _fit(draw, value, font(18), col_w - 16), font=font(18), fill=INK)
    y += 58
    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=RULE, width=2)
    y += 22

    # --- X-ray with findings -----------------------------------------------------------
    y = _draw_xray(page, draw, data, y, max_h=520)
    y = _draw_legend(draw, y + 12)
    y += 22

    # --- Findings, two columns ----------------------------------------------------------
    gap = 40
    col_w = (CONTENT_W - gap) // 2
    left_x, right_x = MARGIN, MARGIN + col_w + gap
    result = data.analysis
    spacing = result.spacing

    yl = _section_title(draw, left_x, y, "FDI Tooth Detection", TOOTH_BOX, col_w)
    status = f"{len(result.detected_numbers)} of 32 teeth detected"
    if spacing.extra_teeth:
        status += f", {spacing.extra_teeth} extra"
    yl = _text_block(draw, left_x, yl, status, col_w, bold=True)
    yl = _draw_chart(draw, left_x, yl + 6, col_w, result) + 14

    yl = _section_title(draw, left_x, yl, "Spacing — missing teeth or spaces", SPACING_BOX, col_w)
    yl = _text_block(draw, left_x, yl, _plural(len(spacing.missing), "missing tooth or space",
                     "missing teeth or spaces") if spacing.missing else "No missing teeth or spaces",
                     col_w, bold=True)
    if spacing.missing:
        yl = _text_block(draw, left_x, yl, f"Missing tooth or space: {', '.join(spacing.missing)}", col_w)
    if spacing.third_molars_missing:
        yl = _text_block(draw, left_x, yl,
                         f"Third molar not detected: {', '.join(spacing.third_molars_missing)}", col_w)

    yr = _section_title(draw, right_x, y, "Overcrowding — extra teeth (possible supernumerary)",
                        CROWDING_BOX, col_w)
    if spacing.has_overcrowding:
        yr = _text_block(draw, right_x, yr, f"Possible overcrowding: "
                         f"{_plural(spacing.extra_teeth, 'extra tooth', 'extra teeth')}", col_w, bold=True)
        for number, count in spacing.extra_numbers.items():
            yr = _text_block(draw, right_x, yr, f"Extra tooth: {count} teeth numbered {number}", col_w)
    else:
        yr = _text_block(draw, right_x, yr, "No extra teeth found", col_w, bold=True)
    yr += 14

    yr = _section_title(draw, right_x, yr, "Impacted Tooth", DANGER, col_w)
    if result.impacted:
        numbers = [i.fdi or "unnumbered" for i in result.impacted]
        yr = _text_block(draw, right_x, yr, f"{_plural(len(numbers), 'impacted tooth', 'impacted teeth')}: "
                         f"{', '.join(numbers)}", col_w, bold=True)
        for item in result.impacted:
            yr = _text_block(draw, right_x, yr,
                             f"{item.fdi or 'Unnumbered'} — {item.detection.score:.0%} confidence", col_w)
    else:
        yr = _text_block(draw, right_x, yr, "No impacted teeth found", col_w, bold=True)

    y = max(yl, yr) + 20
    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=RULE, width=2)
    y += 20

    # --- Clinician verification and notes -------------------------------------------------
    footer_top = PAGE_H - MARGIN - 110
    draw.text((MARGIN, y), "Clinician verification", font=font(22), fill=INK)
    y += 34
    y = _text_block(draw, MARGIN, y, f"Verified and approved by {data.clinician} on "
                    f"{data.approved_at:%d %b %Y} at {data.approved_at:%H:%M}", CONTENT_W, bold=True)
    y += 10
    draw.text((MARGIN, y), "Clinician notes", font=font(16), fill=MUTED)
    y += 26
    box_bottom = footer_top - 20
    draw.rounded_rectangle([MARGIN, y, PAGE_W - MARGIN, box_bottom], radius=10, fill=PANEL, outline=RULE, width=2)
    notes = data.notes.strip() or "No notes."
    _notes_in_box(draw, notes, MARGIN + 18, y + 14, CONTENT_W - 36, box_bottom - 14)

    # --- Footer --------------------------------------------------------------------------
    y = footer_top
    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=RULE, width=1)
    y += 12
    y = _text_block(draw, MARGIN, y, DISCLAIMER, CONTENT_W, size=14, color=MUTED)
    details = (f"Models: {NUMBERING_MODEL.name} (tooth numbering, YOLOv8), {IMPACTED_MODEL.name} "
               f"(impacted teeth, EfficientDet) · OrthoScan AI {data.app_version}".rstrip())
    _text_block(draw, MARGIN, y + 4, details, CONTENT_W, size=13, color=MUTED)
    return page


# --- Layout helpers -----------------------------------------------------------------------
def _draw_xray(page, draw, data, y, max_h):
    """The X-ray with every finding drawn at full colour, centred."""
    fitted = ImageOps.contain(data.image.convert("RGB"), (CONTENT_W, max_h), method=Image.Resampling.LANCZOS)
    fitted = draw_analysis_overlay(
        fitted, data.analysis, scale=fitted.width / data.image.width, tag_size=13, line_scale=2
    )
    x = MARGIN + (CONTENT_W - fitted.width) // 2
    page.paste(fitted, (x, y))
    draw.rectangle([x - 1, y - 1, x + fitted.width, y + fitted.height], outline=RULE, width=1)
    return y + fitted.height


def _draw_legend(draw, y):
    items = [("Tooth (FDI)", TOOTH_BOX, False), ("Missing tooth / space", SPACING_BOX, True),
             ("Extra tooth", CROWDING_BOX, False), ("Impacted tooth", DANGER, False)]
    x = MARGIN
    for label, color, dashed in items:
        if dashed:
            for dx in range(0, 18, 6):
                draw.line([(x + dx, y + 9), (x + dx + 3, y + 9)], fill=color, width=4)
        else:
            draw.rectangle([x, y + 2, x + 16, y + 16], outline=color, width=3)
        draw.text((x + 24, y), label, font=font(15, bold=False), fill=INK)
        x += 24 + int(draw.textlength(label, font=font(15, bold=False))) + 34
    return y + 22


def _section_title(draw, x, y, title, color, width):
    """Colour swatch (matching the X-ray boxes) and a heading that wraps if long."""
    draw.rounded_rectangle([x, y + 6, x + 14, y + 20], radius=3, fill=color)
    for line in _wrap(draw, title, font(20), width - 22):
        draw.text((x + 22, y), line, font=font(20), fill=INK)
        y += 28
    return y + 4


def _text_block(draw, x, y, text, width, bold=False, size=17, color=INK):
    f = font(size, bold)
    for line in _wrap(draw, text, f, width):
        draw.text((x, y), line, font=f, fill=color)
        y += int(size * 1.45)
    return y


def _draw_chart(draw, x, y, width, result):
    """FDI dental chart, coloured like the app: impacted red, extra orange, missing violet."""
    detected = result.detected_numbers
    impacted = result.impacted_numbers
    extra = result.spacing.extra_numbers
    missing = set(result.spacing.missing)
    cell = min(width // 17, 34)
    f = font(15)
    for row, arch in enumerate((UPPER_ARCH, LOWER_ARCH)):
        for i, number in enumerate(arch):
            column = i if i < 8 else i + 1  # column 8 is the midline
            if number in impacted:
                color = DANGER
            elif number in extra:
                color = CROWDING_BOX
            elif number in missing:
                color = SPACING_BOX
            elif number in detected:
                color = INK
            else:
                color = FAINT
            cx = x + column * cell + cell // 2
            cy = y + row * 30 + 15
            draw.text((cx, cy), number, font=f, fill=color, anchor="mm")
    mid_x = x + 8 * cell + cell // 2
    draw.line([(mid_x, y + 2), (mid_x, y + 58)], fill=RULE, width=2)
    draw.line([(x, y + 30), (x + 17 * cell, y + 30)], fill=RULE, width=2)
    return y + 64


def _notes_in_box(draw, text, x, y, width, bottom):
    truncated = len(text) > MAX_NOTES_CHARS
    text = text[:MAX_NOTES_CHARS]
    f = font(17, bold=False)
    line_h = int(17 * 1.45)
    lines = []
    for paragraph in text.splitlines() or [""]:
        lines.extend(_wrap(draw, paragraph, f, width) or [""])
    max_lines = max((bottom - y) // line_h, 2)
    if truncated or len(lines) > max_lines:
        # Keep the marker on its own line so it's always readable.
        lines = lines[: max_lines - 1] + ["… (notes truncated)"]
    for line in lines:
        draw.text((x, y), line, font=f, fill=INK)
        y += line_h


def _wrap(draw, text, f, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}" if line else word
        if line and draw.textlength(trial, font=f) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _fit(draw, text, f, width):
    """Shorten `text` with an ellipsis until it fits `width`."""
    if draw.textlength(text, font=f) <= width:
        return text
    while text and draw.textlength(text + "…", font=f) > width:
        text = text[:-1]
    return text + "…"


def _plural(count, singular, plural):
    return f"{count} {singular if count == 1 else plural}"
