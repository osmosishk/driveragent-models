#!/usr/bin/env python3
"""DriverGuard time split on this device (decode, preprocess, each engine as
H2D/compute/D2H, postprocess, control). Uses the pulled registry version and
its validation frames. Run after `da-models build driverguard`:

    python3 tools/bench_driverguard.py [--model-dir /opt/driveragent/models/driverguard/current]

Note: Jetson DVFS lowers the GPU/CPU clocks when the load is not continuous.
Run `sudo jetson_clocks` first to measure the fixed-clock figures.
"""
import argparse
import json
import math
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--model-dir", default="/opt/driveragent/models/driverguard/current")
_ap.add_argument("--loops", type=int, default=200)
_args = _ap.parse_args()
V = Path(_args.model_dir)
sys.path.insert(0, str(V / "runtime" / "jetson_runtime"))
import onnxruntime as ort  # noqa: E402
import pycuda.driver as cuda  # noqa: E402
from beta_mode import beta_mode_action  # noqa: E402
from preprocess import preprocess_dtcp, preprocess_yolopx  # noqa: E402
from trt_runner import TRTRunner  # noqa: E402
from yolopx_postprocess import nms_yolopx, scale_coords, segmasks_from_logits  # noqa: E402

E = V / "engines"
yolopx = TRTRunner(str(next(E.glob("*/yolopx_v2.engine"))))
dtcp = TRTRunner(str(next(E.glob("*/dtcp_v1_main.engine"))))
ctrl = ort.InferenceSession(str(V / "dtcp_v1_control.onnx"), providers=["CPUExecutionProvider"])
tmp = Path(tempfile.mkdtemp())
with tarfile.open(V / "validation_samples.tar.gz") as t:
    t.extractall(tmp, filter="data")
man = json.loads((tmp / "scene_manifest_subset.json").read_text())
jpg = tmp / "cam_front" / Path(man["image_paths"][0]).name
speed, cmd, target = float(man["speeds_mps"][0]), int(man["commands"][0]), [float(v) for v in man["target_points"][0]]


def infer_split(r, inputs, t):
    """TRTRunner.infer with a sync after each phase: H2D, compute, D2H."""
    for name, arr in inputs.items():
        np.copyto(r.host_buffers[name], np.ascontiguousarray(arr).astype(r.host_buffers[name].dtype, copy=False))
        cuda.memcpy_htod_async(r.device_buffers[name], r.host_buffers[name], r.stream)
    r.stream.synchronize(); t.append(time.perf_counter())
    r.context.execute_async_v3(r.stream.handle)
    r.stream.synchronize(); t.append(time.perf_counter())
    for name in r.output_names:
        cuda.memcpy_dtoh_async(r.host_buffers[name], r.device_buffers[name], r.stream)
    r.stream.synchronize()
    out = {n: r.host_buffers[n].copy() for n in r.output_names}
    t.append(time.perf_counter())
    return out


names = ["decode_jpeg", "pre_yolopx", "yolopx_h2d", "yolopx_compute", "yolopx_d2h", "post_yolopx_nms",
         "post_yolopx_masks", "pre_dtcp", "dtcp_h2d", "dtcp_compute", "dtcp_d2h", "control_ort", "control_beta",
         "post_dtcp"]
acc = {n: [] for n in names}
totals_108 = []  # the stages that the 108 ms loop contained
for i in range(_args.loops + 20):
    t = [time.perf_counter()]
    bgr = cv2.imread(str(jpg)); t.append(time.perf_counter())
    x, h0, w0, pad, ratio = preprocess_yolopx(bgr); t.append(time.perf_counter())
    y = infer_split(yolopx, {"image": x}, t)
    boxes = nms_yolopx(y["det"], conf_thres=0.30, iou_thres=0.45)
    if len(boxes):
        boxes[:, :4] = scale_coords((384, 640), boxes[:, :4], (h0, w0))
    t.append(time.perf_counter())
    segmasks_from_logits(y["da_seg"], y["ll_seg"], (384, 640), pad, (h0, w0)); t.append(time.perf_counter())
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    state = np.concatenate([np.array([[speed / 12.0]], np.float32), np.array([target], np.float32),
                            np.eye(6, dtype=np.float32)[[cmd]]], axis=1)
    dx = preprocess_dtcp(rgb); t.append(time.perf_counter())
    d = infer_split(dtcp, {"image": dx, "state": state, "target_point": np.array([target], np.float32)}, t)
    mu, sg = ctrl.run(None, {"cnn_feature": d["cnn_feature"], "measurement_feature": d["measurement_feature"]})
    t.append(time.perf_counter())
    beta_mode_action(mu[0], sg[0]); t.append(time.perf_counter())
    lat, fwd = float(d["pred_wp"][0, 3, 0]), float(d["pred_wp"][0, 3, 1])
    _ = math.atan(1.78 * 2.0 * lat / (lat * lat + fwd * fwd)) if fwd > 1.0 else 0.0
    _ = float(d["pred_speed"][0, 0]) * 12.0
    t.append(time.perf_counter())
    if i >= 20:
        d_ms = [(b - a) * 1000 for a, b in zip(t[:-1], t[1:])]
        for n, v in zip(names, d_ms):
            acc[n].append(v)
        totals_108.append(sum(v for n, v in zip(names, d_ms)
                              if n not in ("decode_jpeg", "post_yolopx_nms", "post_yolopx_masks", "post_dtcp")))
res = {n: {"mean": round(float(np.mean(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)} for n, v in acc.items()}
res["sum_of_108ms_stages"] = round(float(np.mean(totals_108)), 1)
res["full_frame"] = round(sum(r["mean"] for r in res.values() if isinstance(r, dict)), 1)
res["frame_size"] = [int(h0), int(w0)]
import resource  # noqa: E402
res["process_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
res["rate_hz_full_frame"] = round(1000.0 / res["full_frame"], 2)
print(json.dumps(res, indent=1))
