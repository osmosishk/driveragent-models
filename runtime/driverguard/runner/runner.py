"""DriverGuard inference loop: front cam → YOLOPX + DTCP → ZMQ publish.

Invoked by <version>/runtime/run.py, which puts the driveragent root, the
runtime/jetson_runtime folder, and this package on sys.path first.

Hybrid DTCP inference (v1-fp32-2026-05-25):
  - Main TRT engine: image/state/target -> pred_wp, pred_speed, cnn_feature,
    measurement_feature.
  - Control sub-graph (cnn_feature, measurement_feature -> mu, sigma) runs
    through onnxruntime CPU. TRT 10.3 miscompiles this sub-graph and produces
    collapsed mu/sigma; the ORT path produces values matching PC PyTorch.
"""

import json
import math
import os
import time
import threading

import cv2
import numpy as np
import onnxruntime as ort

from message.capnp_pubsub import (Publisher, Subscriber, DaemonStatus,
                                  DaemonMessenger)

# jetson_bundle/jetson_runtime (on sys.path via run.py)
from trt_runner import TRTRunner
from preprocess import preprocess_yolopx, preprocess_dtcp
from yolopx_postprocess import (nms_yolopx, scale_coords, segmasks_from_logits)
from beta_mode import beta_mode_action

from runner.camera_reader import FrontCameraReader, CAM_WIDTH, CAM_HEIGHT
from runner.ego_state import SpeedProvider
from runner.route_guidance_provider import RouteGuidanceProvider
from runner.mask_codec import encode_rle

SERVICE_NAME = "driverguard_runner"

# DTCP-checkpoint command remap (runner_patches/0001).
# The trained checkpoint only ever saw commands {0:LEFT, 1:RIGHT, 3:LANE_FOLLOW};
# indices 2/4/5 collapse to the trained network output for command 3. Remap
# to the closest trained intent until the model is retrained on a fuller
# command distribution.
#   2 STRAIGHT      -> 3 LANE_FOLLOW
#   4 CHANGE_LEFT   -> 0 LEFT
#   5 CHANGE_RIGHT  -> 1 RIGHT
_CMD_REMAP = {0: 0, 1: 1, 2: 3, 3: 3, 4: 0, 5: 1}

# Vehicle geometry for pred_wp-based steering (runner_patches/0002).
# Keep in sync with control/control_config.ini.
WHEELBASE_M = 1.78          # Citroën Ami wheelbase
MAX_WHEEL_ANGLE_RAD = math.radians(28.0)

# Engine/version tags emitted on every DriverGuardResult.
_ENGINE_PRECISION = "FP32+ort_control"
# <version>/bundle_manifest.json (this file is <version>/runtime/runner/runner.py).
_BUNDLE_MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "bundle_manifest.json")


def _read_model_version(manifest_path=_BUNDLE_MANIFEST):
    try:
        with open(manifest_path) as f:
            return json.load(f).get("version", "unknown")
    except Exception:
        return "unknown"


class SDStatusGate:
    """Background subscriber to SelfDrivingStatus.enabled."""

    def __init__(self, addr, schema_file):
        self._enabled = True
        self._addr = addr
        self._schema_file = schema_file
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="driverguard_sd_gate")
        self._thread.start()

    def _run(self):
        try:
            sub = Subscriber(self._schema_file, "SelfDrivingStatus", self._addr)
        except Exception as e:
            print(f"[driverguard] sd_status gate: cannot subscribe to {self._addr}: "
                  f"{e}. Gate stays open.")
            return
        while not self._stop.is_set():
            try:
                msg = sub.receive()
                self._enabled = bool(getattr(msg, "enabled", True))
            except Exception:
                time.sleep(0.05)

    def is_enabled(self):
        return self._enabled


def _build_state_vec(speed_mps, target_xy, command):
    """Match build_state_vec in jetson_pipeline.py (DTCP dtcp_infer.py:146-150).

    Applies the DTCP-checkpoint command remap (runner_patches/0001) so unknown
    indices collapse to their closest trained intent rather than into LANE_FOLLOW.
    """
    command = _CMD_REMAP.get(int(command), 3)
    speed = np.array([[speed_mps / 12.0]], dtype=np.float32)
    target = np.array([list(target_xy)], dtype=np.float32)
    cmd_one_hot = np.zeros((1, 6), dtype=np.float32)
    cmd_one_hot[0, command] = 1.0
    return np.concatenate([speed, target, cmd_one_hot], axis=1)


def steer_from_pred_wp(pred_wp):
    """Bicycle-model wheel angle from DTCP's t+2 s waypoint (runner_patches/0002).

    Returns normalized steer in [-1, +1] (positive = right). The DTCP `steer`
    scalar is weak even in PC FP32 (spread 0.076 across the entire sweep);
    pred_wp's lateral signal is intact end-to-end (~2 m deflection for a
    ±10 m target), so we drive steer from waypoint curvature.
    """
    lat, fwd = float(pred_wp[3, 0]), float(pred_wp[3, 1])
    if not (math.isfinite(lat) and math.isfinite(fwd)) or fwd <= 1.0:
        return 0.0
    l2 = lat * lat + fwd * fwd
    kappa = 2.0 * lat / l2
    wheel_angle = math.atan(WHEELBASE_M * kappa)
    return max(-1.0, min(1.0, wheel_angle / MAX_WHEEL_ANGLE_RAD))


def _run_one_frame(yolopx, dtcp, ctrl_sess, bgr, speed_mps, target_xy, command):
    """Single forward pass through YOLOPX + DTCP (TRT main + ORT control)."""
    # YOLOPX
    yx, h0, w0, pad_wh, _ratio = preprocess_yolopx(bgr)
    y_out = yolopx.infer({"image": yx})
    boxes = nms_yolopx(y_out["det"], conf_thres=0.30, iou_thres=0.45)
    if len(boxes):
        boxes[:, :4] = scale_coords(yx.shape[-2:], boxes[:, :4], (h0, w0))
    da_mask, ll_mask = segmasks_from_logits(
        y_out["da_seg"], y_out["ll_seg"],
        in_hw=yx.shape[-2:], pad_wh=pad_wh, out_hw=(h0, w0))

    # DTCP main (TRT)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dx = preprocess_dtcp(rgb)
    state = _build_state_vec(speed_mps, target_xy, command)
    target_arr = np.array([list(target_xy)], dtype=np.float32)
    d_out = dtcp.infer({"image": dx, "state": state, "target_point": target_arr})
    wp = d_out["pred_wp"][0]
    pred_speed_mps = float(d_out["pred_speed"][0, 0]) * 12.0

    # DTCP control (ORT CPU) — workaround for TRT 10.3 control-branch miscompile.
    ctrl_out = ctrl_sess.run(None, {
        "cnn_feature": d_out["cnn_feature"],
        "measurement_feature": d_out["measurement_feature"],
    })
    mu = ctrl_out[0][0]
    sigma = ctrl_out[1][0]
    throttle, _steer_unused, brake = beta_mode_action(mu, sigma)
    steer = steer_from_pred_wp(wp)
    return {
        "boxes": boxes,
        "da_mask": da_mask,
        "ll_mask": ll_mask,
        "wp": wp,
        "throttle": float(throttle),
        "steer": float(steer),
        "brake": float(brake),
        "pred_speed_mps": pred_speed_mps,
    }


def main(args):
    control_onnx = getattr(args, "control_onnx", None) or os.path.join(
        os.path.dirname(os.path.abspath(args.dtcp_engine)), "dtcp_v1_control.onnx")
    model_version = _read_model_version()
    print(f"[driverguard] loading engines (model_version={model_version}):\n"
          f"  yolopx:       {args.yolopx_engine}\n"
          f"  dtcp main:    {args.dtcp_engine}\n"
          f"  dtcp control: {control_onnx}")
    yolopx = TRTRunner(args.yolopx_engine)
    dtcp = TRTRunner(args.dtcp_engine)
    ctrl_sess = ort.InferenceSession(control_onnx, providers=["CPUExecutionProvider"])
    print(f"[driverguard] {yolopx}")
    print(f"[driverguard] {dtcp}")

    try:
        daemon_status = DaemonStatus(name=SERVICE_NAME)
        daemon_log = DaemonMessenger(name=SERVICE_NAME)
    except Exception as e:
        print(f"[driverguard] heartbeat unavailable: {e}")
        daemon_status = None
        daemon_log = None

    camera = FrontCameraReader(socket_path=args.cam)
    camera.connect(max_retries=10)

    speed_provider = SpeedProvider()
    speed_provider.start()

    route_guidance = None
    if not args.no_route_guidance:
        route_guidance = RouteGuidanceProvider(addr=args.route_guidance_addr)
        route_guidance.start()
        print(f"[driverguard] route guidance: subscribing to "
              f"{args.route_guidance_addr}")

    gate = None
    if args.gated:
        gate = SDStatusGate(args.sd_status_addr, args.schema)
        gate.start()

    pub_addr = f"tcp://*:{args.pub_port}"
    pub = Publisher(args.schema, "DriverGuardResult", pub_addr, bind=True)
    print(f"[driverguard] publishing DriverGuardResult on {pub_addr}")

    target_xy = tuple(args.target)
    command = int(args.command)
    period = 1.0 / max(args.rate, 0.1)
    frame_idx = 0

    try:
        while True:
            tick_start = time.perf_counter()
            t_now_ms = int(time.time() * 1000)

            if gate is not None and not gate.is_enabled():
                time.sleep(period)
                continue

            bgr = camera.read_bgr()
            if bgr is None:
                if daemon_log is not None:
                    try:
                        daemon_log.log("waiting for camera frame")
                    except Exception:
                        pass
                time.sleep(0.05)
                continue
            if bgr.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
                bgr = cv2.resize(bgr, (CAM_WIDTH, CAM_HEIGHT),
                                 interpolation=cv2.INTER_LINEAR)

            speed_mps = speed_provider.latest_speed_mps()

            cur_command = command
            cur_target = target_xy
            if route_guidance is not None:
                g_cmd, g_target, g_fresh = route_guidance.latest_guidance()
                if g_fresh:
                    cur_command = g_cmd
                    if g_target is not None:
                        cur_target = g_target

            t0 = time.perf_counter()
            try:
                result = _run_one_frame(yolopx, dtcp, ctrl_sess, bgr, speed_mps,
                                        cur_target, cur_command)
            except Exception as e:
                print(f"[driverguard] inference failed: {e}")
                if daemon_status is not None:
                    try:
                        daemon_status.send(False, f"inference: {e}")
                    except Exception:
                        pass
                time.sleep(period)
                continue
            inference_ms = (time.perf_counter() - t0) * 1000.0

            wp = result["wp"]
            trajectory = [{"x": float(wp[i, 0]), "y": float(wp[i, 1])}
                          for i in range(wp.shape[0])]
            finite_mask = np.isfinite(wp)
            finite = bool(finite_mask.all())
            if not finite:
                bad = np.argwhere(~finite_mask)
                _msg = (f"[wp-debug] frame={frame_idx} "
                        f"non-finite wp at idx={bad.tolist()}  "
                        f"raw={wp.tolist()}  "
                        f"any_nan={bool(np.isnan(wp).any())}  "
                        f"any_inf={bool(np.isinf(wp).any())}")
                print(_msg, flush=True)
                try:
                    with open("/tmp/wp_debug.log", "a") as _f:
                        _f.write(_msg + "\n")
                except Exception:
                    pass

            try:
                da_rle = encode_rle(result["da_mask"])
                ll_rle = encode_rle(result["ll_mask"])
            except Exception as e:
                print(f"[driverguard] mask encode failed: {e}")
                da_rle = b""
                ll_rle = b""

            detections = []
            for row in result["boxes"]:
                detections.append({
                    "x1": float(row[0]),
                    "y1": float(row[1]),
                    "x2": float(row[2]),
                    "y2": float(row[3]),
                    "conf": float(row[4]),
                    "classId": int(row[5]) & 0xFF,
                })

            try:
                pub.publish(
                    timestamp=t_now_ms,
                    frame=frame_idx,
                    trajectory=trajectory,
                    throttle=result["throttle"],
                    steer=result["steer"],
                    brake=result["brake"],
                    predSpeedMps=result["pred_speed_mps"],
                    egoSpeedMps=float(speed_mps),
                    command=cur_command,
                    inferenceMs=float(inference_ms),
                    finite=finite,
                    daMaskRle=da_rle,
                    llMaskRle=ll_rle,
                    maskWidth=CAM_WIDTH,
                    maskHeight=CAM_HEIGHT,
                    detections=detections,
                    enginePrecision=_ENGINE_PRECISION,
                    modelVersion=model_version,
                )
            except Exception as e:
                print(f"[driverguard] publish failed: {e}")

            if daemon_status is not None and frame_idx % 10 == 0:
                try:
                    daemon_status.send(
                        True,
                        f"frame={frame_idx} ms={inference_ms:.1f} "
                        f"det={len(detections)}")
                except Exception:
                    pass

            if frame_idx % 10 == 0:
                print(f"[driverguard] frame={frame_idx} {inference_ms:.1f} ms "
                      f"dets={len(detections)} cmd={cur_command} "
                      f"t/s/b=({result['throttle']:.2f},{result['steer']:+.2f},"
                      f"{result['brake']:.2f}) "
                      f"wp4=({wp[-1, 0]:+.1f},{wp[-1, 1]:+.1f}) "
                      f"da_rle={len(da_rle)}B ll_rle={len(ll_rle)}B "
                      f"speed={speed_mps:.1f} m/s")

            frame_idx += 1
            if args.once:
                break

            elapsed = time.perf_counter() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("[driverguard] interrupted, shutting down")
    finally:
        camera.close()
        speed_provider.stop()
        if route_guidance is not None:
            route_guidance.stop()
        if daemon_status is not None:
            try:
                daemon_status.send(False, "exiting")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(
        "Do not run runner/runner.py directly. "
        "Use: python <version>/runtime/run.py")
