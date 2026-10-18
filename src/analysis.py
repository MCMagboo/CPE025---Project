"""Rules on model detections: FDI numbers for impacted teeth, and spacing/overcrowding.

No ML and no UI code, so it's quick to unit-test. Boxes are (x1, y1, x2, y2)
in original-image pixels.

Spacing & overcrowding come from the tooth numbering, not from measurements:
more teeth than the FDI numbering allows means possible overcrowding, and a
number with no tooth means a missing tooth or a space.
"""

from collections import Counter
from dataclasses import dataclass, field
from statistics import median

# FDI numbers in the order they sit along each jaw on a panoramic X-ray.
UPPER_ARCH = [str(n) for n in [*range(18, 10, -1), *range(21, 29)]]
LOWER_ARCH = [str(n) for n in [*range(48, 40, -1), *range(31, 39)]]
ALL_TEETH = UPPER_ARCH + LOWER_ARCH
THIRD_MOLARS = {"18", "28", "38", "48"}

MATCH_IOU = 0.3  # impacted box must overlap a tooth box this much to take its number
# The numbering model sometimes puts two neighbouring numbers on one tooth (boxes
# overlapping 0.77-0.98 on sample X-rays). At or above this overlap, boxes are one
# tooth. 0.7 is also Ultralytics' own NMS default.
SAME_TOOTH_IOU = 0.7
# A gap box between neighbours that touch is widened to this fraction of a tooth, so it stays visible.
MIN_SPACE_WIDTH = 0.5


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class ImpactedTooth:
    detection: Detection
    fdi: str | None  # None when it doesn't line up with a numbered tooth


@dataclass(frozen=True)
class SpacingRegion:
    numbers: tuple  # consecutive missing FDI numbers sharing this gap, e.g. ("44", "45")
    box: tuple  # (x1, y1, x2, y2) between the neighbouring teeth


@dataclass(frozen=True)
class SpacingAssessment:
    # FDI number -> how many separate teeth carry it (only if > 1). A quadrant has
    # just 8 numbers, so any 9th tooth in it necessarily repeats one: this alone
    # catches every tooth beyond the numbering.
    extra_numbers: dict
    missing: list  # numbers with no tooth (missing tooth or space), third molars excluded
    third_molars_missing: list  # 18/28/38/48 with no tooth; often absent naturally
    spaces: list = field(default_factory=list)  # SpacingRegion per gap that has teeth on both sides

    @property
    def extra_teeth(self):
        return sum(count - 1 for count in self.extra_numbers.values())

    @property
    def has_overcrowding(self):
        return bool(self.extra_numbers)


@dataclass
class AnalysisResult:
    teeth: list
    impacted: list
    spacing: SpacingAssessment
    timings_ms: dict = field(default_factory=dict)

    @property
    def detected_numbers(self):
        return {t.label for t in self.teeth}

    @property
    def impacted_numbers(self):
        return {i.fdi for i in self.impacted if i.fdi}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_same_spot(detections, threshold=SAME_TOOTH_IOU):
    """One box per physical tooth: of boxes overlapping >= threshold, keep the most confident.

    Labels are ignored on purpose, so a tooth labelled both 12 and 13 counts once.
    Separate teeth with the same number are kept (that's an extra tooth).
    """
    kept = []
    for det in sorted(detections, key=lambda d: -d.score):
        if all(iou(det.box, k.box) < threshold for k in kept):
            kept.append(det)
    return kept


def assign_fdi(impacted, teeth):
    """Give each impacted box the FDI number of the tooth box it overlaps most."""
    result = []
    for det in sorted(impacted, key=lambda d: -d.score):
        best = max(teeth, key=lambda t: iou(det.box, t.box), default=None)
        fdi = best.label if best is not None and iou(det.box, best.box) >= MATCH_IOU else None
        result.append(ImpactedTooth(det, fdi))
    return result


def assess_spacing(teeth):
    """Overcrowding and missing teeth from the numbering (teeth already merged per spot)."""
    per_number = Counter(t.label for t in teeth)
    missing = [n for n in sorted(ALL_TEETH, key=int) if n not in per_number]
    missing_spaces = [n for n in missing if n not in THIRD_MOLARS]
    return SpacingAssessment(
        extra_numbers={
            n: c for n, c in sorted(per_number.items(), key=lambda kv: int(kv[0])) if c > 1
        },
        missing=missing_spaces,
        third_molars_missing=[n for n in missing if n in THIRD_MOLARS],
        spaces=spacing_regions(teeth, set(missing_spaces)),
    )


def spacing_regions(teeth, missing):
    """Where each missing tooth should be: the gap between its nearest detected neighbours.

    Consecutive missing numbers in a jaw share one region. A group with no
    detected tooth on one side (the end of the jaw) gets no region.
    """
    best = {}
    for tooth in teeth:  # a repeated number uses its most confident box
        if tooth.label not in best or tooth.score > best[tooth.label].score:
            best[tooth.label] = tooth

    regions = []
    for arch in (UPPER_ARCH, LOWER_ARCH):
        present = [best[n] for n in arch if n in best]
        if not present:
            continue
        min_width = median(t.box[2] - t.box[0] for t in present) * MIN_SPACE_WIDTH
        i = 0
        while i < len(arch):
            if arch[i] not in missing:
                i += 1
                continue
            start = i
            while i < len(arch) and arch[i] in missing:
                i += 1
            group = tuple(sorted(arch[start:i], key=int))  # label reads "44, 45"
            before = next((best[n] for n in reversed(arch[:start]) if n in best), None)
            after = next((best[n] for n in arch[i:] if n in best), None)
            if before is not None and after is not None:
                regions.append(SpacingRegion(group, _gap_box(before, after, min_width)))
    return regions


def _gap_box(a, b, min_width):
    left, right = sorted((a, b), key=lambda t: t.box[0] + t.box[2])
    x1, x2 = left.box[2], right.box[0]
    if x2 - x1 < min_width:  # neighbours touch or overlap: centre a minimum-width box
        middle = (x1 + x2) / 2
        x1, x2 = middle - min_width / 2, middle + min_width / 2
    y1 = (left.box[1] + right.box[1]) / 2
    y2 = (left.box[3] + right.box[3]) / 2
    return (x1, y1, x2, y2)
