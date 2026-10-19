"""Drawing the AI findings onto an X-ray image (Pillow only, no UI code).

Shared by the app's canvas (main.py) and the exported PDF report (report.py),
so both show the boxes the same way.
"""

from PIL import Image, ImageColor, ImageDraw, ImageFont

from theme import BG_SIDEBAR, CROWDING_BOX, DANGER, SPACING_BOX, TOOTH_BOX

DIMMED_ALPHA = 60  # ~25% opacity for boxes of findings that aren't highlighted

_fonts = {}


def font(size, bold=True):
    """Segoe UI at `size` px (falls back to Pillow's built-in font)."""
    key = (size, bold)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)
        except OSError:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


def draw_dashed_rect(draw, box, color, width=2, dash=14, gap=10):
    """Dashed rectangle; `box` must be whole pixels."""
    left, top, right, bottom = box
    for x in range(left, right, dash + gap):
        x_end = min(x + dash, right)
        draw.line([(x, top), (x_end, top)], fill=color, width=width)
        draw.line([(x, bottom), (x_end, bottom)], fill=color, width=width)
    for y in range(top, bottom, dash + gap):
        y_end = min(y + dash, bottom)
        draw.line([(left, y), (left, y_end)], fill=color, width=width)
        draw.line([(right, y), (right, y_end)], fill=color, width=width)


def draw_tag(draw, box, text, color, tag_font, text_color=BG_SIDEBAR):
    """Small filled label at a box's top-left corner (inside if there's no room above)."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=tag_font)
    width, height = right - left + 6, bottom - top + 4
    x = box[0]
    y = box[1] - height if box[1] - height >= 0 else box[1]
    draw.rectangle([x, y, x + width, y + height], fill=color)
    draw.text((x + 3 - left, y + 2 - top), text, fill=text_color, font=tag_font)


def draw_analysis_overlay(image, analysis, scale, highlighted=frozenset(), tag_size=10, line_scale=1):
    """AI boxes on a copy of `image`, one colour per finding (as on the assessment cards):
    blue teeth, orange extra teeth, dashed violet gaps, thick red impacted teeth on top.

    Findings not in `highlighted` are faded; an empty set highlights everything.
    `tag_size` / `line_scale` enlarge labels and lines (e.g. for a printed report).
    Returns a new RGB image.
    """
    impacted = analysis.impacted_numbers
    extra = analysis.spacing.extra_numbers
    # (finding, dashed, box, tag, colour, line width), in drawing order.
    shapes = []
    for tooth in analysis.teeth:
        if tooth.label in impacted:
            continue  # drawn in red below
        is_extra = tooth.label in extra
        shapes.append((
            "crowding" if is_extra else "fdi", False, [c * scale for c in tooth.box],
            tooth.label, CROWDING_BOX if is_extra else TOOTH_BOX, 2 if is_extra else 1,
        ))
    for space in analysis.spacing.spaces:
        shapes.append((
            "spacing", True, [round(c * scale) for c in space.box],
            f"Space {', '.join(space.numbers)}", SPACING_BOX, 2,
        ))
    for item in analysis.impacted:
        shapes.append((
            "impacted", False, [c * scale for c in item.detection.box],
            f"Impacted {item.fdi}" if item.fdi else "Impacted", DANGER, 3,
        ))

    def bright(finding):
        return not highlighted or finding in highlighted

    # Draw on a transparent layer, faded findings first so they never cover bright ones.
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    tag_font = font(tag_size)
    for finding, dashed, box, tag, color, width in sorted(shapes, key=lambda s: bright(s[0])):
        alpha = 255 if bright(finding) else DIMMED_ALPHA
        rgba = (*ImageColor.getrgb(color), alpha)
        width *= line_scale
        if dashed:
            draw_dashed_rect(draw, box, rgba, width=width, dash=5 * line_scale, gap=3 * line_scale)
        else:
            draw.rectangle(box, outline=rgba, width=width)
        draw_tag(draw, box, tag, rgba, tag_font, text_color=(*ImageColor.getrgb(BG_SIDEBAR), alpha))
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
