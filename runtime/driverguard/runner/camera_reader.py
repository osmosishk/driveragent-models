"""Single-camera GStreamer reader for /tmp/cam0 (RGBA 1280x720 @ 30fps).

Returns BGR uint8 frames so downstream consumers (the jetson_bundle's
preprocess_yolopx / preprocess_dtcp helpers) can use them unchanged.
"""

import os
import time

import cv2
import numpy as np

CAM_WIDTH = 1280
CAM_HEIGHT = 720


def _build_pipeline(socket_path):
    return (
        f"shmsrc socket-path={socket_path} is-live=true do-timestamp=true ! "
        f"video/x-raw,format=RGBA,width={CAM_WIDTH},height={CAM_HEIGHT},"
        f"framerate=30/1 ! "
        f"queue max-size-buffers=2 leaky=downstream ! "
        f"appsink sync=false max-buffers=1 drop=true"
    )


class FrontCameraReader:
    def __init__(self, socket_path="/tmp/cam0"):
        self.socket_path = socket_path
        self.cap = None
        self.last_frame_bgr = None
        self._last_open_attempt = 0.0
        self._reopen_interval = 1.0

    def _try_open(self):
        if not os.path.exists(self.socket_path):
            return False
        try:
            old_level = os.environ.get("GST_DEBUG", "")
            os.environ["GST_DEBUG"] = "0"
            cap = cv2.VideoCapture(_build_pipeline(self.socket_path),
                                   cv2.CAP_GSTREAMER)
            if old_level:
                os.environ["GST_DEBUG"] = old_level
            elif "GST_DEBUG" in os.environ:
                del os.environ["GST_DEBUG"]
            if cap.isOpened():
                self.cap = cap
                return True
            cap.release()
        except Exception:
            pass
        return False

    def connect(self, max_retries=10, retry_delay=1.0):
        for _ in range(max_retries):
            if self._try_open():
                print(f"[driverguard] camera connected: {self.socket_path}")
                return True
            time.sleep(retry_delay)
        print(f"[driverguard] WARNING: camera {self.socket_path} not connected; "
              f"will retry on read")
        return False

    def read_bgr(self):
        """Returns BGR uint8 [720, 1280, 3] or None if no frame is available."""
        now = time.time()
        if self.cap is None:
            if now - self._last_open_attempt > self._reopen_interval:
                self._last_open_attempt = now
                self._try_open()
            return self.last_frame_bgr  # may be None on first call

        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None
        if not ret or frame is None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            return self.last_frame_bgr

        if frame.ndim == 3 and frame.shape[2] == 4:
            # OpenCV's GStreamer caps return BGRA-style 4ch; convert to BGR.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.last_frame_bgr = frame
        return frame

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
