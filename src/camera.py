"""Webcam / document-camera ingestion for OrthoScan AI.

No UI code here. CameraStream opens the camera and reads frames on a
background thread, keeping only the newest frame, so the UI can poll it at
its own pace without freezing. Frames stay in memory; nothing is saved.
"""

import threading
import time

import cv2
from PIL import Image

# Sharper for reading a printed X-ray; cameras fall back to what they support.
REQUESTED_SIZE = (1920, 1080)
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
        self._frame = None  # newest BGR frame from OpenCV
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

    def latest_frame(self, max_size=None):
        """Newest frame as an RGB PIL image, or None if there isn't one yet.

        With `max_size=(w, h)`, the frame is shrunk to fit first, which is much
        cheaper than converting the full-resolution frame for a live preview.
        """
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        if max_size is not None:
            h, w = frame.shape[:2]
            ratio = min(max_size[0] / w, max_size[1] / h, 1.0)
            if ratio < 1.0:
                size = (max(int(w * ratio), 1), max(int(h * ratio), 1))
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
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
            # Uncompressed 1080p is too much for USB (~4 fps); MJPG keeps it smooth.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_SIZE[1])

            failures = 0
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
                with self._lock:
                    self._frame = frame
                    self.frame_count += 1
                if self.state == self.STARTING:
                    self.frame_size = (frame.shape[1], frame.shape[0])
                    self.state = self.RUNNING
        finally:
            cap.release()

    def _fail(self, message):
        self.error = message
        self.state = self.FAILED


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
