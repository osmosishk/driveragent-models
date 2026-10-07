"""6-camera GStreamer reader for /tmp/cam0..cam5 (RGBA 1280x720 @ 30fps).

Adapted from /home/tonyho/model/sparsedrive/run/camera_integration.py.
Returns a torch.uint8 tensor [6, 3, H, W] in RGB order (alpha dropped).
"""

import os
import time
import argparse

import cv2
import numpy as np

CAM_WIDTH = 1280
CAM_HEIGHT = 720
NUM_CAMERAS = 6


class GStreamerCameraReader:
    def __init__(self, cam_dir="/tmp", num_cameras=NUM_CAMERAS):
        self.cam_dir = cam_dir
        self.num_cameras = num_cameras
        self._caps = [None] * num_cameras
        self._last_frame = [None] * num_cameras
        self._fail_count = [0] * num_cameras
        self._last_reopen = [0.0] * num_cameras
        self._reopen_interval = 1.0
        self._fail_threshold = 10
        self._connected = False

    def _build_pipeline(self, cam_index):
        sock_path = os.path.join(self.cam_dir, f"cam{cam_index}")
        return (
            f"shmsrc socket-path={sock_path} ! "
            f"video/x-raw,format=RGBA,width={CAM_WIDTH},height={CAM_HEIGHT},"
            f"framerate=30/1 ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )

    def _open_one(self, i):
        cap = cv2.VideoCapture(self._build_pipeline(i), cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._caps[i] = cap
            return True
        cap.release()
        return False

    def connect(self, max_retries=10, retry_delay=1.0):
        remaining = set(range(self.num_cameras))
        for attempt in range(max_retries):
            failed = set()
            for i in sorted(remaining):
                sock_path = os.path.join(self.cam_dir, f"cam{i}")
                if not os.path.exists(sock_path):
                    failed.add(i)
                    continue
                try:
                    if not self._open_one(i):
                        failed.add(i)
                except Exception:
                    failed.add(i)
            remaining = failed
            if not remaining:
                break
            time.sleep(retry_delay)
        self._connected = True
        connected = sum(1 for c in self._caps if c is not None)
        print(f"[camera_reader] {connected}/{self.num_cameras} cameras connected")
        return connected == self.num_cameras

    def _close_one(self, i):
        if self._caps[i] is not None:
            try:
                self._caps[i].release()
            except Exception:
                pass
            self._caps[i] = None

    def _try_reopen(self, i, now):
        if now - self._last_reopen[i] < self._reopen_interval:
            return
        self._last_reopen[i] = now
        sock_path = os.path.join(self.cam_dir, f"cam{i}")
        if os.path.exists(sock_path) and self._open_one(i):
            self._fail_count[i] = 0

    def read_frames(self):
        """Returns np.ndarray [6, H, W, 3] uint8 RGB, or None if all cameras failed."""
        if not self._connected:
            return None
        frames = []
        any_valid = False
        now = time.time()
        for i in range(self.num_cameras):
            cap = self._caps[i]
            frame = None
            if cap is not None:
                ret, frame = cap.read()
                if ret and frame is not None:
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    self._last_frame[i] = frame
                    self._fail_count[i] = 0
                    any_valid = True
                else:
                    self._fail_count[i] += 1
                    frame = self._last_frame[i]
                    if self._fail_count[i] >= self._fail_threshold:
                        self._close_one(i)
                        self._try_reopen(i, now)
            else:
                frame = self._last_frame[i]
                self._try_reopen(i, now)
            if frame is None:
                frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            frames.append(frame)
        if not any_valid and all(f is None for f in self._last_frame):
            return None
        return np.stack(frames)

    def read_frames_torch(self):
        """Convenience: returns torch.uint8 [6, 3, H, W] in RGB or None."""
        import torch
        arr = self.read_frames()
        if arr is None:
            return None
        # arr: [6, H, W, 3] uint8 → [6, 3, H, W]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        return t

    def close(self):
        for i in range(self.num_cameras):
            self._close_one(i)


def _probe():
    """python -m system1_runner.camera_reader --probe"""
    reader = GStreamerCameraReader()
    if not reader.connect(max_retries=5):
        print("[probe] WARNING: not all cameras connected")
    # Wait for first frames
    for _ in range(20):
        arr = reader.read_frames()
        if arr is not None:
            break
        time.sleep(0.1)
    if arr is None:
        print("[probe] FAIL: no frames after 2s")
        return 1
    print(f"[probe] frames shape: {arr.shape}, dtype: {arr.dtype}")
    for i in range(arr.shape[0]):
        f = arr[i]
        print(f"  cam{i}: mean={f.mean():.1f} std={f.std():.1f} "
              f"nonzero={(f != 0).any()}")
    reader.close()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    if args.probe:
        raise SystemExit(_probe())
    else:
        p.print_help()
