#!/usr/bin/env python3
"""Bundle parity check — go/no-go gate before swapping the engine on the live vehicle.

Two modes, distinguished by --capture-pc-reference:

  1) --capture-pc-reference  (run on the SERVER, DTCP env)
     Runs the original PC PyTorch DTCPPlanner on the 3 frames in
     samples/cam_front/ using the manifest's per-frame speed/command/target_point,
     writes samples/reference_dtcp_pc.json — the golden actions the Jetson must
     reproduce.

  2) (default, run on the JETSON)
     Loads the freshly-built TRT engine via jetson_runtime/jetson_pipeline.py
     primitives, runs the same 3 frames, compares to reference_dtcp_pc.json.
     Hard-fails if throttle MAE > 0.02, steer MAE > 0.02, or finite is False
     on any frame.

This is the smallest possible parity surface (3 frames, 6 scalars per frame)
but it nails the parity-report symptoms: FP16 collapse zeros throttle, FP32
restores it. If this passes, the new engine is safe to deploy.

Usage (server, capture):
    PYTHONPATH=/home/tonyho/development/DTCP/deploy \\
      /home/tonyho/anaconda3/envs/DTCP/bin/python \\
      /home/tonyho/development/jetson_bundle/validate_jetson.py \\
      --capture-pc-reference

Usage (Jetson, validate):
    PYTHONPATH=~/jetson_bundle/jetson_runtime python \\
      ~/jetson_bundle/validate_jetson.py \\
      --dtcp-engine   ~/jetson_bundle/engines/dtcp_v1_fp32.engine \\
      --yolopx-engine ~/jetson_bundle/engines/yolopx_v2_fp16.engine
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
SAMPLES = BUNDLE / "samples"
DEFAULT_REF = SAMPLES / "reference_dtcp_pc.json"
DEFAULT_MANIFEST = SAMPLES / "scene_manifest_subset.json"
DEFAULT_FRAMES_DIR = SAMPLES / "cam_front"

# Hard-fail thresholds — match MANIFEST.json "sample_thresholds".
THRESH_THROTTLE_MAE = 0.02
THRESH_STEER_MAE = 0.02


def _load_manifest():
    m = json.loads(DEFAULT_MANIFEST.read_text())
    frames = sorted(DEFAULT_FRAMES_DIR.glob("*.jpg"))
    if len(frames) != len(m["image_paths"]):
        raise SystemExit(f"sample frame count mismatch: dir={len(frames)} manifest={len(m['image_paths'])}")
    name_to_idx = {Path(p).name: i for i, p in enumerate(m["image_paths"])}
    rows = []
    for f in frames:
        if f.name not in name_to_idx:
            raise SystemExit(f"frame {f.name} not in manifest")
        k = name_to_idx[f.name]
        rows.append({
            "frame_name": f.name,
            "frame_path": str(f),
            "speed_mps": float(m["speeds_mps"][k]),
            "command":   int(m["commands"][k]),
            "target_xy": [float(v) for v in m["target_points"][k]],
        })
    return rows


def capture_pc_reference(out_path: Path):
    """Run DTCPPlanner (PC PyTorch) and dump golden actions."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, "/home/tonyho/development/DTCP/deploy")
    from dtcp_infer import DTCPPlanner  # noqa: E402

    weights = BUNDLE / "weights" / "dtcp_nusc_route_v1.pt"
    if not weights.exists():
        raise SystemExit(f"missing weights: {weights}")
    planner = DTCPPlanner(weights=str(weights), device="cuda")

    rows = _load_manifest()
    ref = {
        "source": "PC PyTorch DTCPPlanner (jetson_bundle/weights/dtcp_nusc_route_v1.pt)",
        "frames": [],
    }
    print(f"capturing PC reference for {len(rows)} frames")
    for r in rows:
        img = np.array(Image.open(r["frame_path"]).convert("RGB"))
        out = planner.infer(img, r["speed_mps"], tuple(r["target_xy"]), r["command"])
        ref["frames"].append({
            "frame_name": r["frame_name"],
            "speed_mps":  r["speed_mps"],
            "command":    r["command"],
            "target_xy":  r["target_xy"],
            "throttle":   float(out["throttle"]),
            "steer":      float(out["steer"]),
            "brake":      float(out["brake"]),
            "pred_speed_mps": float(out["pred_speed_mps"])
                if "pred_speed_mps" in out else float(out.get("pred_speed", 0.0)),
            "mu":    [float(v) for v in out["mu"]],
            "sigma": [float(v) for v in out["sigma"]],
            "wp4":   [float(out["waypoints"][-1, 0]), float(out["waypoints"][-1, 1])],
        })
        print(f"  {r['frame_name']}  thr={out['throttle']:.4f}  steer={out['steer']:+.4f}  brake={out['brake']:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ref, indent=2))
    print(f"wrote {out_path}")


def validate_on_jetson(dtcp_engine: str, yolopx_engine: str, ref_path: Path,
                       control_onnx: str = None):
    """Run hybrid TRT+ORT pipeline on the 3 sample frames and compare to PC reference.

    TRT 10.3 miscompiles the DTCP control sub-graph (mu/sigma collapse).
    Workaround: split the model into a main TRT engine (pred_wp/pred_speed/
    cnn_feature/measurement_feature) and a control ONNX (cnn_feature,
    measurement_feature -> mu, sigma) run via onnxruntime CPU.
    """
    import cv2
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(BUNDLE / "jetson_runtime"))
    from trt_runner import TRTRunner
    from preprocess import preprocess_dtcp, preprocess_yolopx
    from beta_mode import beta_mode_action

    if control_onnx is None:
        control_onnx = str(BUNDLE / "onnx" / "dtcp_v1_control.onnx")

    if not ref_path.exists():
        raise SystemExit(f"reference missing: {ref_path} — run with --capture-pc-reference on the server first")
    ref = json.loads(ref_path.read_text())
    ref_by_name = {r["frame_name"]: r for r in ref["frames"]}

    print(f"loading dtcp main engine: {dtcp_engine}")
    dtcp = TRTRunner(dtcp_engine)
    print(f"loading dtcp control onnx: {control_onnx}")
    ctrl_sess = ort.InferenceSession(control_onnx, providers=["CPUExecutionProvider"])
    if yolopx_engine and Path(yolopx_engine).exists():
        print(f"loading yolopx engine: {yolopx_engine}")
        _ = TRTRunner(yolopx_engine)

    rows = _load_manifest()
    print(f"\n{'frame':<55} {'PC thr':>8} {'TRT thr':>8} {'Δ':>8} "
          f"{'PC steer':>9} {'TRT steer':>9} {'Δ':>8} {'finite':>7}")

    thr_errs, steer_errs = [], []
    all_finite = True
    fail_lines = []
    for r in rows:
        if r["frame_name"] not in ref_by_name:
            raise SystemExit(f"reference has no entry for {r['frame_name']}")
        g = ref_by_name[r["frame_name"]]

        bgr = cv2.imread(r["frame_path"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dx = preprocess_dtcp(rgb)
        speed = np.array([[r["speed_mps"] / 12.0]], dtype=np.float32)
        target = np.array([list(r["target_xy"])], dtype=np.float32)
        cmd_oh = np.zeros((1, 6), dtype=np.float32); cmd_oh[0, r["command"]] = 1.0
        state = np.concatenate([speed, target, cmd_oh], axis=1)
        d_out = dtcp.infer({"image": dx, "state": state, "target_point": target})
        ctrl_out = ctrl_sess.run(None, {
            "cnn_feature": d_out["cnn_feature"],
            "measurement_feature": d_out["measurement_feature"],
        })
        mu, sigma = ctrl_out[0][0], ctrl_out[1][0]
        thr, steer, brake = beta_mode_action(mu, sigma)
        finite = bool(np.isfinite([thr, steer, brake]).all()
                      and np.isfinite(d_out["pred_wp"]).all())

        d_thr = abs(thr - g["throttle"])
        d_steer = abs(steer - g["steer"])
        thr_errs.append(d_thr); steer_errs.append(d_steer)
        if not finite:
            all_finite = False
            fail_lines.append(f"  finite=False on {r['frame_name']}")
        print(f"{r['frame_name']:<55} "
              f"{g['throttle']:>+8.4f} {thr:>+8.4f} {d_thr:>8.4f} "
              f"{g['steer']:>+9.4f} {steer:>+9.4f} {d_steer:>8.4f} "
              f"{str(finite):>7}")

    mae_thr = float(np.mean(thr_errs))
    mae_steer = float(np.mean(steer_errs))
    print(f"\n  throttle MAE: {mae_thr:.4f}  (threshold {THRESH_THROTTLE_MAE:.4f})")
    print(f"  steer    MAE: {mae_steer:.4f}  (threshold {THRESH_STEER_MAE:.4f})")
    print(f"  finite==True on all: {all_finite}")

    failed = []
    if mae_thr > THRESH_THROTTLE_MAE:
        failed.append(f"throttle MAE {mae_thr:.4f} > {THRESH_THROTTLE_MAE:.4f}")
    if mae_steer > THRESH_STEER_MAE:
        failed.append(f"steer MAE {mae_steer:.4f} > {THRESH_STEER_MAE:.4f}")
    if not all_finite:
        failed.extend(fail_lines)

    if failed:
        print("\nFAIL")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-pc-reference", action="store_true",
                    help="run PC PyTorch DTCPPlanner and write samples/reference_dtcp_pc.json")
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--dtcp-engine", default=None,
                    help="TRT engine path; default ~/jetson_bundle/engines/dtcp_v1_fp32.engine")
    ap.add_argument("--yolopx-engine", default=None,
                    help="optional YOLOPX engine to load-check; default ~/jetson_bundle/engines/yolopx_v2_fp16.engine")
    ap.add_argument("--control-onnx", default=None,
                    help="control sub-graph ONNX path; default <bundle>/onnx/dtcp_v1_control.onnx")
    args = ap.parse_args()

    if args.capture_pc_reference:
        capture_pc_reference(Path(args.reference))
        return 0

    dtcp = args.dtcp_engine or str(Path.home() / "jetson_bundle/engines/dtcp_v1_fp32.engine")
    yolopx = args.yolopx_engine or str(Path.home() / "jetson_bundle/engines/yolopx_v2_fp16.engine")
    return validate_on_jetson(dtcp, yolopx, Path(args.reference), control_onnx=args.control_onnx)


if __name__ == "__main__":
    sys.exit(main())
