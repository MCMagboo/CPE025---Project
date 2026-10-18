"""AI inference for OrthoScan AI: tooth numbering and impacted-tooth detection.

No UI code. Analyzer.load() is slow and runs once, off the UI thread;
analyze() then takes well under a second on CPU. Heavy packages (torch,
ultralytics) are imported only inside load(), so the app still starts in a
Python without them and can explain what's missing.

Models (in models/):
  tooth_numbering.pt - Ultralytics YOLOv8s, 32 FDI classes, trained at 1024 px.
  impacted_tooth.pt  - TorchScript EfficientDet, 1 class, 640x640 input; it
                       normalizes internally and needs torchvision for NMS.
"""

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from analysis import AnalysisResult, Detection, assess_spacing, assign_fdi, merge_same_spot
from preprocessing import clahe, letterbox, to_rgb

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NUMBERING_MODEL = MODELS_DIR / "tooth_numbering.pt"
IMPACTED_MODEL = MODELS_DIR / "impacted_tooth.pt"

NUMBERING_IMGSZ = 1024  # the size it was trained at (stored in the model)
IMPACTED_SIZE = 640
# Tuned on sample X-rays: real findings scored 0.34-0.98, noise at most ~0.2.
NUMBERING_CONF = 0.30
IMPACTED_CONF = 0.30
# Tested on sample X-rays: CLAHE lowered confidence for both models, so it's off.
USE_CLAHE = False

UNAVAILABLE_HELP = (
    "AI models need Python 3.13 with the packages in requirements.txt "
    "(run: py -3.13 src\\main.py)."
)


class AnalysisUnavailable(Exception):
    """The models can't run; the message is shown to the clinician."""


class ToothNumberingModel:
    def __init__(self, path=NUMBERING_MODEL):
        from ultralytics import YOLO

        self.model = YOLO(str(path))

    def predict(self, rgb):
        result = self.model.predict(
            rgb, imgsz=NUMBERING_IMGSZ, conf=NUMBERING_CONF, device="cpu", verbose=False
        )[0]
        boxes = result.boxes
        detections = [
            Detection(self.model.names[int(cls)], float(score), tuple(box))
            for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
        # One box per physical tooth. Same-number teeth on separate spots are kept:
        # they're extra teeth, which the spacing assessment reports as overcrowding.
        return merge_same_spot(detections)


class ImpactedToothModel:
    def __init__(self, path=IMPACTED_MODEL):
        import torch
        import torchvision  # noqa: F401 - registers torchvision::nms used by the scripted model

        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        # TorchScript optimizes on its first runs (~1.5 s each); do that now, not on a real X-ray.
        warmup = Image.new("RGB", (IMPACTED_SIZE, IMPACTED_SIZE))
        for _ in range(2):
            self._run(warmup)

    def predict(self, rgb):
        # Letterbox beat stretching on sample X-rays (higher scores, less noise).
        square, scale = letterbox(rgb, IMPACTED_SIZE)
        boxes, scores = self._run(square)
        width, height = rgb.size
        found = []
        for (x1, y1, x2, y2), score in zip(boxes / scale, scores):
            if score < IMPACTED_CONF:
                continue
            box = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            found.append(Detection("impacted", float(score), box))
        return found

    def _run(self, square):
        torch = self.torch
        pixels = np.asarray(square, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        with torch.inference_mode():
            out = self.model([tensor])[0]
        return out["boxes"].numpy(), out["scores"].numpy()


class Analyzer:
    """Both models behind one call. Not thread-safe: use from one worker thread."""

    def __init__(self):
        self.numbering = None
        self.impacted = None

    @property
    def loaded(self):
        return self.numbering is not None and self.impacted is not None

    def load(self):
        if self.loaded:
            return
        for path in (NUMBERING_MODEL, IMPACTED_MODEL):
            if not path.is_file():
                raise AnalysisUnavailable(
                    f"Model file missing: models/{path.name}. Put it back in the models folder."
                )
        try:
            _configure_ultralytics()
            numbering = ToothNumberingModel()
            impacted = ImpactedToothModel()
        except ImportError as exc:
            raise AnalysisUnavailable(UNAVAILABLE_HELP) from exc
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the clinician
            raise AnalysisUnavailable(f"The AI models couldn't be loaded ({exc}).") from exc
        self.numbering, self.impacted = numbering, impacted

    def analyze(self, image):
        self.load()
        rgb = to_rgb(image)
        if USE_CLAHE:
            rgb = clahe(rgb)
        start = time.perf_counter()
        teeth = self.numbering.predict(rgb)
        after_numbering = time.perf_counter()
        impacted = self.impacted.predict(rgb)
        end = time.perf_counter()
        return AnalysisResult(
            teeth=sorted(teeth, key=lambda t: t.label),
            impacted=assign_fdi(impacted, teeth),
            spacing=assess_spacing(teeth),
            timings_ms={
                "numbering": (after_numbering - start) * 1000,
                "impacted": (end - after_numbering) * 1000,
                "total": (end - start) * 1000,
            },
        )


def _configure_ultralytics():
    """Keep Ultralytics fully offline: no usage analytics, no downloads, quiet logs."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import settings

    if settings.get("sync"):
        settings.update({"sync": False})
