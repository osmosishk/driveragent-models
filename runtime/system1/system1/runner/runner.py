"""System1 main runner: cameras → preprocess → inference → ZMQ publish.

Invoked through /home/tonyho/model/system1/run.py, which puts both the system1
root and the driveragent root on sys.path before calling main(args).
"""

import os
import sys
import time
import threading

import numpy as np
import torch
import torch.nn.functional as F

from message.capnp_pubsub import Publisher, Subscriber, DaemonStatus, DaemonMessenger

from runner.camera_reader import GStreamerCameraReader
from runner.calibration import load_all as load_calibration
from runner.ego_state import EgoStateProvider, CMD_STRAIGHT

DEFAULT_SCHEMA = "/home/tonyho/driveragent/message/message.capnp"
DEFAULT_PUB_ADDR = "tcp://*:8011"
DEFAULT_SD_STATUS_ADDR = "tcp://127.0.0.1:5595"
DEFAULT_CHECKPOINT = "/home/tonyho/model/system1/system1_deploy.pth"
SERVICE_NAME = "system1_runner"


def _import_system1():
    """Import the model loader from run_system1.py (sibling of this package).
    The caller (run.py) is responsible for putting that dir on sys.path."""
    try:
        from run_system1 import load_model, run_inference, preprocess_images
    except ImportError as e:
        msg = (
            f"Cannot import run_system1: {e}\n"
            f"Expected /home/tonyho/model/system1/run_system1.py on sys.path.\n"
            f"Per DEPLOY_INSTRUCTIONS.md Step 2, copy these from the training server:\n"
            f"  system1/{{__init__.py, config.py, system1_model.py, scorer/...}}\n"
            f"  models_convnext/backbone.py\n"
            f"  models/{{backbone.py, fpn.py}}\n"
        )
        raise RuntimeError(msg) from e
    return load_model, run_inference, preprocess_images


class SDStatusGate:
    """Background subscriber to SelfDrivingStatus.enabled. If the topic is
    unreachable the gate stays open (so a missing UI doesn't block testing)."""
    def __init__(self, addr=DEFAULT_SD_STATUS_ADDR, schema_file=DEFAULT_SCHEMA):
        self._enabled = True
        self._stop = threading.Event()
        self._addr = addr
        self._schema_file = schema_file
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="sd_status_gate")
        self._thread.start()

    def _run(self):
        try:
            sub = Subscriber(self._schema_file, "SelfDrivingStatus", self._addr)
        except Exception as e:
            print(f"[sd_status] WARNING: cannot subscribe at {self._addr}: {e}. "
                  f"Gate will remain open.")
            return
        while not self._stop.is_set():
            try:
                msg = sub.receive()
                self._enabled = bool(getattr(msg, "enabled", True))
            except Exception:
                time.sleep(0.05)

    def is_enabled(self):
        return self._enabled


class AsyncCameraReader:
    """Background thread that keeps the latest 6-camera frame stack ready,
    so the inference loop never blocks on cap.read()."""
    def __init__(self, reader):
        self._reader = reader
        self._latest = None
        self._latest_ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="system1_camera")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            arr = self._reader.read_frames()
            if arr is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = arr
                self._latest_ts = time.time()

    def latest(self):
        with self._lock:
            return self._latest, self._latest_ts

    def stop(self):
        self._stop.set()


_PREPROC_MEAN = None
_PREPROC_STD = None


def _gpu_preprocess(imgs_np, preproc_cfg, device):
    """[6, H, W, 3] uint8 numpy → [1, 6, 3, 256, 704] float on GPU.
    Matches run_system1.preprocess_images: scale to (raw*resize), top-left crop
    to final_size, ImageNet normalize. Done entirely on GPU."""
    global _PREPROC_MEAN, _PREPROC_STD
    if _PREPROC_MEAN is None:
        _PREPROC_MEAN = torch.tensor(preproc_cfg['mean'], device=device).view(1, 3, 1, 1)
        _PREPROC_STD  = torch.tensor(preproc_cfg['std'],  device=device).view(1, 3, 1, 1)

    H_raw, W_raw = preproc_cfg['raw_size']
    H_final, W_final = preproc_cfg['final_size']
    scale = preproc_cfg['resize']
    new_h = int(H_raw * scale)
    new_w = int(W_raw * scale)

    # Upload uint8 first (cheaper transfer than float), then cast on GPU.
    x = torch.from_numpy(imgs_np).to(device, non_blocking=True)        # [6,H,W,3]
    x = x.permute(0, 3, 1, 2).contiguous().float() / 255.0             # [6,3,H,W]
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear',
                      align_corners=False, antialias=False)
    crop_x = max(0, (new_w - W_final) // 2)
    x = x[:, :, 0:H_final, crop_x:crop_x + W_final]
    x = (x - _PREPROC_MEAN) / _PREPROC_STD
    return x.unsqueeze(0)


def _flatten_traj(traj_t):
    """traj_t: torch.Tensor [6, 3] (x, y, heading) → (list[Point2D-dicts], list[float])."""
    pts = []
    headings = []
    for i in range(traj_t.shape[0]):
        x = float(traj_t[i, 0])
        y = float(traj_t[i, 1])
        h = float(traj_t[i, 2])
        pts.append({"x": x, "y": y})
        headings.append(h)
    return pts, headings


def main(args):
    """Run the system1 inference loop. `args` is an argparse.Namespace built
    by run.py (see DEFAULT_* constants for the field set)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[system1_runner] device={device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Heartbeat / log channels (best-effort; don't fail startup if hub isn't up)
    try:
        daemon_status = DaemonStatus(name=SERVICE_NAME)
        daemon_log = DaemonMessenger(name=SERVICE_NAME)
    except Exception as e:
        print(f"[system1_runner] heartbeat unavailable: {e}")
        daemon_status = None
        daemon_log = None

    # Model
    load_model, run_inference, preprocess_images = _import_system1()
    print(f"[system1_runner] loading {args.checkpoint} ...")
    model, preproc_cfg = load_model(args.checkpoint, device=device)

    # Calibration (1280x720 stream → 704x256 model input)
    proj_mat_cpu, image_wh_cpu = load_calibration()
    proj_mat = proj_mat_cpu.to(device)
    image_wh = image_wh_cpu.to(device)

    # Cameras (read in a background thread so the inference loop never blocks
    # on cv2.cap.read() — sequential reads of 6 cams can otherwise add 30-60ms)
    reader = None
    async_cams = None
    if not args.no_cameras:
        reader = GStreamerCameraReader()
        reader.connect(max_retries=10)
        async_cams = AsyncCameraReader(reader)
        async_cams.start()

    # Ego state subscriber
    ego = EgoStateProvider()
    ego.start()

    # Optional gate
    gate = None
    if args.gated:
        gate = SDStatusGate(addr=args.sd_status_addr, schema_file=args.schema)
        gate.start()

    # Publisher
    pub_addr = f"tcp://*:{args.pub_port}"
    pub = Publisher(args.schema, "System1Result", pub_addr, bind=True)
    print(f"[system1_runner] publishing System1Result on {pub_addr}")

    # Preproc tensors that are constant across frames
    preproc_cfg_local = dict(preproc_cfg)
    # Override raw_size to match the actual camera stream so the resize math
    # in preprocess_images matches what the calibration loader assumed.
    preproc_cfg_local["raw_size"] = (720, 1280)

    period = 1.0 / max(args.rate, 0.1)
    frame_idx = 0
    try:
        while True:
            tick_start = time.perf_counter()
            t_now_ms = int(time.time() * 1000)

            if gate is not None and not gate.is_enabled():
                time.sleep(period)
                continue

            # Acquire 6 frames from the background reader
            if async_cams is not None:
                imgs_np, frame_ts = async_cams.latest()
                if imgs_np is None:
                    if daemon_log is not None:
                        try:
                            daemon_log.log("waiting for camera frames")
                        except Exception:
                            pass
                    time.sleep(0.05)
                    continue
            else:
                imgs_np = np.zeros((6, 720, 1280, 3), dtype=np.uint8)

            try:
                images = _gpu_preprocess(imgs_np, preproc_cfg_local, device=device)
            except Exception as e:
                print(f"[system1_runner] preprocess failed: {e}")
                time.sleep(period)
                continue

            ego_state = ego.build_ego_state(cmd=CMD_STRAIGHT)

            t0 = time.perf_counter()
            try:
                trajectory, _ = run_inference(
                    model, images, ego_state, proj_mat, image_wh,
                    scene_ctx=None, device=device, precision=args.precision,
                )
            except Exception as e:
                print(f"[system1_runner] inference failed: {e}")
                if daemon_status is not None:
                    try:
                        daemon_status.send(False, f"inference: {e}")
                    except Exception:
                        pass
                time.sleep(period)
                continue
            inference_ms = (time.perf_counter() - t0) * 1000

            finite = bool(torch.isfinite(trajectory).all().item())
            pts, headings = _flatten_traj(trajectory)

            try:
                pub.publish(
                    timestamp=t_now_ms,
                    frame=frame_idx,
                    trajectory=pts,
                    headings=headings,
                    egoSpeedKph=float(ego.speed_kph()),
                    cmd=0,
                    inferenceMs=float(inference_ms),
                    finite=finite,
                )
            except Exception as e:
                print(f"[system1_runner] publish failed: {e}")

            if daemon_status is not None and frame_idx % 10 == 0:
                try:
                    daemon_status.send(True, f"frame={frame_idx} ms={inference_ms:.1f}")
                except Exception:
                    pass

            if frame_idx % 10 == 0:
                first = pts[0]
                last = pts[-1]
                print(f"[system1_runner] frame={frame_idx} {inference_ms:.1f} ms "
                      f"finite={finite} t1=({first['x']:.2f},{first['y']:.2f}) "
                      f"t6=({last['x']:.2f},{last['y']:.2f}) "
                      f"speed={ego.speed_kph():.1f} kph")

            frame_idx += 1
            if args.once:
                break

            elapsed = time.perf_counter() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("[system1_runner] interrupted, shutting down")
    finally:
        if async_cams is not None:
            async_cams.stop()
        if reader is not None:
            reader.close()
        ego.stop()
        if daemon_status is not None:
            try:
                daemon_status.send(False, "exiting")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(
        "Do not run runner/runner.py directly. "
        "Use: python /home/tonyho/model/system1/run.py [--checkpoint ...]"
    )
