"""Webcam / document-camera ingestion for OrthoScan AI.

No UI code here. CameraStream opens the camera and reads frames on a
background thread, keeping only the newest frame, so the UI can poll it at
its own pace without freezing. Frames stay in memory; nothing is saved.
"""

import threading
import time

import cv2
from PIL import Image

# 720p, because most webcams only reach 60 fps at this size (1080p usually tops
# out at 30). Cameras fall back to the nearest mode they support.
REQUESTED_SIZE = (1280, 720)
REQUESTED_FPS = 60
# Smoothing for the measured frame rate (closer to 1 = steadier readout).
FPS_SMOOTHING = 0.95
# Consecutive failed reads (about 1.5 s) before the camera counts as lost.
MAX_READ_FAILURES = 30

try:
    # Probing camera indices that don't exist makes OpenCV print warnings.
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def _open_capture(index):
    # DirectShow opens much faster than the default backend on Windows.
    return cv2.VideoCapture(index, cv2.CAP_DSHOW)


class CameraStream:
    """One camera, read continuously on a daemon thread."""

    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

    def __init__(self, index):
        self.index = index
        self.state = self.STOPPED
        self.error = None
        self.frame_size = None
        self.frame_count = 0
        self.fps = 0.0  # measured frames per second, 0 until two frames arrive
        self._frame = None  # newest BGR frame from OpenCV
        self._preview = None  # newest frame fitted to _preview_size, as RGB
        self._preview_size = None  # (w, h) in screen pixels, set by the UI
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def name(self):
        return camera_name(self.index)

    def start(self):
        if self._thread is not None:
            return
        self.state = self.STARTING
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{self.index}", daemon=True
        )
        self._thread.start()

    def stop(self, wait=False):
        """Ask the reader thread to stop; it releases the camera itself.

        `wait=True` blocks until the camera is released (used on exit and in
        tests). The UI doesn't wait, because opening can still be in progress.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if wait and thread is not None:
            thread.join(timeout=3)
        if self.state != self.FAILED:
            self.state = self.STOPPED

    def set_preview_size(self, size):
        """Fit future previews into `size` (w, h); None stops making previews."""
        self._preview_size = size

    def latest_preview(self):
        """Newest frame fitted to the preview size, as an RGB numpy array, or None.

        Resizing and colour conversion happen on the reader thread, so the UI
        thread only has to display it.
        """
        with self._lock:
            return self._preview

    def latest_frame(self):
        """Newest full-resolution frame as an RGB PIL image, or None if there isn't one yet."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _run(self):
        cap = _open_capture(self.index)
        try:
            if not cap.isOpened():
                self._fail(
                    f"{self.name} couldn't be opened. Close other apps using it "
                    "(Teams, Zoom, Camera), or pick another camera."
                )
                return
            # Size first, then MJPG: on DirectShow, setting the size after the format
            # can drop back to uncompressed YUY2, which USB can't carry at 60 fps.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_SIZE[1])
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

            failures = 0
            interval = None  # smoothed seconds between frames
            last_time = None
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self._fail(
                            f"{self.name} stopped sending pictures. Check that it's "
                            "connected, or pick another camera."
                        )
                        return
                    time.sleep(0.05)
                    continue
                failures = 0
                now = time.perf_counter()
                if last_time is not None:
                    gap = now - last_time
                    interval = gap if interval is None else (
                        FPS_SMOOTHING * interval + (1 - FPS_SMOOTHING) * gap
                    )
                    self.fps = 1 / interval if interval > 0 else 0.0
                last_time = now

                preview = _fit_rgb(frame, self._preview_size)
                with self._lock:
                    self._frame = frame
                    self._preview = preview
                    self.frame_count += 1
                if self.state == self.STARTING:
                    self.frame_size = (frame.shape[1], frame.shape[0])
                    self.state = self.RUNNING
        finally:
            cap.release()

    def _fail(self, message):
        self.error = message
        self.state = self.FAILED


def _fit_rgb(frame, size):
    """BGR frame scaled to fit `size` (w, h) without stretching, as RGB; None if no size."""
    if size is None:
        return None
    h, w = frame.shape[:2]
    ratio = min(size[0] / w, size[1] / h)
    if ratio != 1.0:
        scaled = (max(int(w * ratio), 1), max(int(h * ratio), 1))
        interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, scaled, interpolation=interpolation)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def camera_name(index):
    # OpenCV can't read device names on Windows, so cameras are numbered from 1.
    return f"Camera {index + 1}"


def list_cameras(max_index=4, skip=()):
    """Indices of cameras that open, probing 0..max_index. Blocking; run off the UI thread.

    Indices in `skip` aren't probed (e.g. the camera already open, which a
    second open could fail on).
    """
    found = []
    for index in range(max_index + 1):
        if index in skip:
            continue
        cap = _open_capture(index)
        try:
            if cap.isOpened():
                found.append(index)
        finally:
            cap.release()
    return found
