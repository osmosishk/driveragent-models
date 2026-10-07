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
    python validate_jetson.py --capture-pc-reference \\
      --dtcp-source <dir with dtcp_infer.py> --weights <dtcp_nusc_route_v1.pt> \\
      --samples <samples dir> --reference <samples dir>/reference_dtcp_pc.json

Usage (Jetson, validate; registry layout from da-models):
    python <version>/runtime/validate_jetson.py
    # defaults: engines from <version>/engines/<tag>/, control ONNX, reference
    # and samples (validation_samples.tar.gz) from <version>/
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # <version>/runtime
MODEL_DIR = HERE.parent                          # <version>
DEFAULT_SAMPLES = MODEL_DIR / "validation_samples.tar.gz"
DEFAULT_REF = MODEL_DIR / "reference_dtcp_pc.json"
# Set by use_samples(): a samples dir with scene_manifest_subset.json + cam_front/.
DEFAULT_MANIFEST = None
DEFAULT_FRAMES_DIR = None


def use_samples(path: Path):
    """Point the loader at a samples dir, or at a .tar.gz of one (extracted to a temp dir)."""
    global DEFAULT_MANIFEST, DEFAULT_FRAMES_DIR
    path = Path(path)
    if path.is_file():
        tmp = Path(tempfile.mkdtemp(prefix="dg-samples-"))
        with tarfile.open(path) as t:
            t.extractall(tmp, filter="data")
        path = tmp
    if not (path / "scene_manifest_subset.json").is_file():
        raise SystemExit(f"no scene_manifest_subset.json in {path}")
    DEFAULT_MANIFEST = path / "scene_manifest_subset.json"
    DEFAULT_FRAMES_DIR = path / "cam_front"


def _find_engine(name):
    hits = sorted((MODEL_DIR / "engines").glob(f"*/{name}"))
    return str(hits[0]) if len(hits) == 1 else None

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


def capture_pc_reference(out_path: Path, dtcp_source: str, weights: str):
    """Run DTCPPlanner (PC PyTorch) and dump golden actions."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, dtcp_source)
    from dtcp_infer import DTCPPlanner  # noqa: E402

    weights = Path(weights)
    if not weights.exists():
        raise SystemExit(f"missing weights: {weights}")
    planner = DTCPPlanner(weights=str(weights), device="cuda")

    rows = _load_manifest()
    ref = {
        "source": f"PC PyTorch DTCPPlanner ({weights.name})",
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
    sys.path.insert(0, str(HERE / "jetson_runtime"))
    from trt_runner import TRTRunner
    from preprocess import preprocess_dtcp, preprocess_yolopx
    from beta_mode import beta_mode_action

    if control_onnx is None:
        control_onnx = str(MODEL_DIR / "dtcp_v1_control.onnx")

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
                    help="run PC PyTorch DTCPPlanner and write the reference JSON")
    ap.add_argument("--dtcp-source", help="capture only: dir that contains dtcp_infer.py")
    ap.add_argument("--weights", help="capture only: DTCP PyTorch weights (dtcp_nusc_route_v1.pt)")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES),
                    help="samples dir or .tar.gz; default <version>/validation_samples.tar.gz")
    ap.add_argument("--reference", default=str(DEFAULT_REF))
    ap.add_argument("--dtcp-engine", default=None,
                    help="TRT engine path; default <version>/engines/<tag>/dtcp_v1_main.engine")
    ap.add_argument("--yolopx-engine", default=None,
                    help="optional YOLOPX engine to load-check; default <version>/engines/<tag>/yolopx_v2.engine")
    ap.add_argument("--control-onnx", default=None,
                    help="control sub-graph ONNX path; default <version>/dtcp_v1_control.onnx")
    args = ap.parse_args()
    use_samples(Path(args.samples))

    if args.capture_pc_reference:
        if not args.dtcp_source or not args.weights:
            raise SystemExit("--capture-pc-reference needs --dtcp-source and --weights")
        capture_pc_reference(Path(args.reference), args.dtcp_source, args.weights)
        return 0

    dtcp = args.dtcp_engine or _find_engine("dtcp_v1_main.engine")
    yolopx = args.yolopx_engine or _find_engine("yolopx_v2.engine")
    if not dtcp:
        raise SystemExit("no DTCP engine: run `da-models build driverguard` or give --dtcp-engine")
    return validate_on_jetson(dtcp, yolopx, Path(args.reference), control_onnx=args.control_onnx)


if __name__ == "__main__":
    sys.exit(main())
