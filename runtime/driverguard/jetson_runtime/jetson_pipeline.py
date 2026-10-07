"""Combined DTCP + YOLOPX inference on Jetson, using TRT engines.

Mirrors the PC two-stage pipeline (run_yolopx_stage.py + run_dtcp_and_viz.py)
but runs both models in one process from TensorRT engines instead of two
PyTorch envs.

Inputs per frame:
    - BGR image from disk or live camera (e.g. 1600x900 nuScenes-equivalent)
    - speed_mps (float)
    - target_point (lateral_right_pos, forward) in metres, ego frame
    - command (int 0..5)

Outputs:
    - Per-frame 3-panel composite (front-cam overlay | BEV | text) as PNG
    - Optional MP4 if processing a folder/video

Usage:
    python jetson_pipeline.py \\
        --yolopx-engine ~/engines/yolopx_v2_fp16.engine \\
        --dtcp-engine   ~/engines/dtcp_v1_fp16.engine \\
        --frames-dir    ~/samples/cam_front \\
        --manifest      ~/samples/scene_manifest_subset.json \\
        --out-dir       ~/out
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from trt_runner import TRTRunner
from preprocess import preprocess_dtcp, preprocess_yolopx
from yolopx_postprocess import nms_yolopx, scale_coords, segmasks_from_logits, CLASS_NAMES
from beta_mode import beta_mode_action
from viz_helpers import (waypoints_to_bev, project_wp_to_image,
                         blend_seg_masks, draw_box, color_for_class)

CMD_NAMES = {0: "LEFT", 1: "RIGHT", 2: "STRAIGHT", 3: "LANE_FOLLOW",
             4: "CHANGE_LEFT", 5: "CHANGE_RIGHT"}


def overlay_panel1(img_rgb, da_mask, ll_mask, boxes, wp_bev):
    out = blend_seg_masks(img_rgb, da_mask, ll_mask, alpha=0.45)
    if len(boxes):
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2, conf, cls in boxes:
            cls_i = int(cls)
            name = CLASS_NAMES[cls_i] if cls_i < len(CLASS_NAMES) else f"cls{cls_i}"
            draw_box(bgr, (x1, y1, x2, y2),
                     label=f"{name} {conf:.2f}",
                     color_bgr=color_for_class(cls_i),
                     line_thickness=2)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = out.shape[:2]
    pix = project_wp_to_image(wp_bev, h, w)
    for i, (u, v) in enumerate(pix):
        if not np.isfinite(u):
            continue
        cu, cv_ = int(round(u)), int(round(v))
        if 0 <= cu < w and 0 <= cv_ < h:
            cv2.circle(out, (cu, cv_), 8, (255, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(out, (cu, cv_), 8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, str(i + 1), (cu - 5, cv_ + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def render_three_panel(panel1_rgb, wp_bev, speed_in, command, target_xy,
                       throttle, brake, steer, speed_pred, boxes,
                       frame_idx, n_frames):
    fig = plt.figure(figsize=(18, 6), dpi=100)
    gs = GridSpec(1, 3, width_ratios=[3, 1.4, 1.0], wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(panel1_rgb)
    ax1.set_title(f"frame {frame_idx + 1}/{n_frames}", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter([0], [0], c="black", marker="^", s=200, zorder=3, label="ego")
    ax2.plot(wp_bev[:, 0], wp_bev[:, 1], "o-", c="tab:red", markersize=10, lw=2, label="DTCP wp")
    for i, (lat, fwd) in enumerate(wp_bev):
        ax2.annotate(f"{i+1}", (lat, fwd), textcoords="offset points", xytext=(7, 7), fontsize=10)
    ax2.scatter([target_xy[0]], [target_xy[1]], c="tab:blue", marker="*", s=180, zorder=3, label="target")
    ax2.set_xlim(-15, 15); ax2.set_ylim(-2, 35)
    ax2.set_xlabel("lateral (m, +right)"); ax2.set_ylabel("forward (m)")
    ax2.set_aspect("equal", adjustable="box"); ax2.grid(True, ls=":", alpha=0.5)
    ax2.set_title("BEV (ego frame)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis("off")
    cls_counter = Counter(int(c) for c in boxes[:, 5]) if len(boxes) else Counter()
    det_summary = "  ".join(f"{CLASS_NAMES[c]}:{n}" for c, n in cls_counter.most_common(5)) or "(none)"
    cmd_name = CMD_NAMES.get(int(command), str(command))
    txt = (
        "INPUT\n"
        f"  speed:   {speed_in:5.2f} m/s\n"
        f"  command: {cmd_name}\n"
        f"  target:  ({target_xy[0]:+5.1f},{target_xy[1]:+5.1f}) m\n"
        "\n"
        "ACTION\n"
        f"  throttle: {throttle:5.3f}\n"
        f"  brake:    {brake:5.3f}\n"
        f"  steer:    {steer:+5.3f}\n"
        "\n"
        "DRIVER GUARD PREDICTION\n"
        f"  next-speed: {speed_pred:5.2f} m/s\n"
        f"  wp4 fwd:    {wp_bev[-1, 1]:5.2f} m\n"
        f"  wp4 lat:    {wp_bev[-1, 0]:+5.2f} m\n"
        "\n"
        f"DETECTION ({len(boxes)})\n  {det_summary}"
    )
    ax3.text(0.0, 0.99, txt, transform=ax3.transAxes, fontsize=11, family="monospace", va="top")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def build_state_vec(speed_mps: float, target_xy: tuple, command: int) -> np.ndarray:
    """Match DTCP/deploy/dtcp_infer.py:146-150."""
    speed = np.array([[speed_mps / 12.0]], dtype=np.float32)
    target = np.array([list(target_xy)], dtype=np.float32)
    cmd_one_hot = np.zeros((1, 6), dtype=np.float32)
    cmd_one_hot[0, command] = 1.0
    return np.concatenate([speed, target, cmd_one_hot], axis=1)


def run_one_frame(yolopx, dtcp, bgr, speed, target_xy, command):
    # YOLOPX preprocess
    yx, h0, w0, pad_wh, _ = preprocess_yolopx(bgr)
    y_out = yolopx.infer({"image": yx})
    boxes = nms_yolopx(y_out["det"], conf_thres=0.30, iou_thres=0.45)
    if len(boxes):
        boxes[:, :4] = scale_coords(yx.shape[-2:], boxes[:, :4], (h0, w0))
    da_mask, ll_mask = segmasks_from_logits(y_out["da_seg"], y_out["ll_seg"],
                                            in_hw=yx.shape[-2:], pad_wh=pad_wh,
                                            out_hw=(h0, w0))
    # DTCP preprocess + infer
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dx = preprocess_dtcp(rgb)
    state = build_state_vec(speed, target_xy, command)
    target = np.array([list(target_xy)], dtype=np.float32)
    d_out = dtcp.infer({"image": dx, "state": state, "target_point": target})

    wp = d_out["pred_wp"][0]                # (4, 2)
    mu = d_out["mu"][0]
    sigma = d_out["sigma"][0]
    pred_speed_mps = float(d_out["pred_speed"][0, 0]) * 12.0
    throttle, steer, brake = beta_mode_action(mu, sigma)
    return rgb, boxes, da_mask, ll_mask, wp, throttle, steer, brake, pred_speed_mps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolopx-engine", required=True)
    ap.add_argument("--dtcp-engine", required=True)
    ap.add_argument("--frames-dir", required=True,
                    help="Folder of JPGs; processed in sorted order")
    ap.add_argument("--manifest",
                    help="Optional JSON with per-frame speeds_mps/commands/target_points "
                         "(same schema as combined_inference/scene_manifest.json). "
                         "If omitted, defaults are used.")
    ap.add_argument("--default-speed", type=float, default=5.0)
    ap.add_argument("--default-command", type=int, default=3)
    ap.add_argument("--default-target", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--mp4", action="store_true", help="Also write combined.mp4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.frames_dir)
    jpgs = sorted(frames_dir.glob("*.jpg"))
    if not jpgs:
        raise SystemExit(f"no .jpg in {frames_dir}")

    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        speeds = m["speeds_mps"]
        commands = m["commands"]
        targets = m["target_points"]
        # Align by filename basename if image_paths in manifest don't match jpgs dir.
        name_to_idx = {Path(p).name: i for i, p in enumerate(m["image_paths"])}
    else:
        speeds = commands = targets = None
        name_to_idx = None

    print(f"loading engines:\n  yolopx: {args.yolopx_engine}\n  dtcp:   {args.dtcp_engine}")
    yolopx = TRTRunner(args.yolopx_engine)
    dtcp = TRTRunner(args.dtcp_engine)
    print(yolopx); print(dtcp)

    writer = None
    t0 = time.time()
    for i, jpg in enumerate(jpgs):
        bgr = cv2.imread(str(jpg))
        if bgr is None:
            print(f"  SKIP {jpg.name}: failed to load"); continue

        if name_to_idx is not None and jpg.name in name_to_idx:
            k = name_to_idx[jpg.name]
            speed = float(speeds[k]); cmd = int(commands[k])
            target_xy = tuple(float(v) for v in targets[k])
        else:
            speed = args.default_speed
            cmd = args.default_command
            target_xy = tuple(args.default_target)

        rgb, boxes, da, ll, wp, thr, steer, brake, speed_pred = run_one_frame(
            yolopx, dtcp, bgr, speed, target_xy, cmd)
        wp_bev = waypoints_to_bev(wp)

        panel1 = overlay_panel1(rgb, da, ll, boxes, wp_bev)
        composite = render_three_panel(panel1, wp_bev, speed, cmd, target_xy,
                                       thr, brake, steer, speed_pred, boxes,
                                       frame_idx=i, n_frames=len(jpgs))
        Image.fromarray(composite).save(out_dir / f"frame_{i:04d}.png")

        if args.mp4:
            bgr_out = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
            if writer is None:
                h, w = bgr_out.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "combined.mp4"),
                                          cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(bgr_out)
        dt = time.time() - t0
        print(f"  [{i+1}/{len(jpgs)}] {jpg.name}  "
              f"t/s/b=({thr:.2f},{steer:+.2f},{brake:.2f})  "
              f"wp4=({wp_bev[-1, 0]:+.1f},{wp_bev[-1, 1]:+.1f})  "
              f"dets={len(boxes)}  ({dt:.1f}s)")

    if writer is not None:
        writer.release()
    print(f"\n[done] {len(jpgs)} frames in {time.time()-t0:.1f}s, output: {out_dir}")


if __name__ == "__main__":
    main()
