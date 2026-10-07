"""Stand-alone vendored helpers for the stage-2 fused viz.

Kept dependency-free of YOLOPX so it runs in the DTCP env. Includes:
  - waypoints_to_bev          (no-op for route-trained DTCP, kept for clarity)
  - project_wp_to_image       (flat-ground pinhole projection, nominal nuScenes intrinsics)
  - blend_seg_masks           (alpha-blend drivable + lane masks onto an RGB image)
  - draw_box                  (rectangle + class label, like YOLOPX/lib/utils/plot.plot_one_box)

Per DTCP/deployment.md caveat #6: nominal CAM_FRONT intrinsics are
fx = fy = 1266, cx = 816, cy = 491, mount at (1.72 m forward, 1.49 m up).
"""
from __future__ import annotations

import cv2
import numpy as np

# Nominal nuScenes CAM_FRONT intrinsics (deployment.md caveat #6).
NOMINAL_FX = 1266.0
NOMINAL_FY = 1266.0
NOMINAL_CX = 816.0
NOMINAL_CY = 491.0
CAM_HEIGHT_M = 1.49           # mount height above ground
CAM_FORWARD_OFFSET_M = 1.72   # ego-to-cam forward offset


# Distinct per-class BGR colours (10 BDD classes), close to the YOLOPX demo style.
CLASS_COLORS_BGR = [
    (255,  64,  64),  # person      — blue-ish red
    (255, 128,   0),  # rider       — orange
    ( 64, 255,  64),  # car         — green
    ( 64, 200, 255),  # bus         — light blue
    (200,   0, 255),  # truck       — magenta
    (  0, 255, 255),  # bike        — yellow
    (255, 255,   0),  # motor       — cyan
    (255,   0, 128),  # traffic light
    (128, 255, 128),  # traffic sign
    (200, 200, 200),  # train
]


def waypoints_to_bev(wp: np.ndarray) -> np.ndarray:
    """Convert raw DTCP waypoints to BEV (lateral_right, forward).

    The route-trained model already uses (col 0 = lateral right-positive,
    col 1 = forward positive), so this is a no-op kept for naming clarity.
    """
    return wp.copy()


def project_wp_to_image(wp_bev: np.ndarray, img_h: int, img_w: int,
                        cam_height_m: float = CAM_HEIGHT_M,
                        cam_forward_offset_m: float = CAM_FORWARD_OFFSET_M,
                        fx: float = NOMINAL_FX, fy: float = NOMINAL_FY,
                        cx: float = NOMINAL_CX, cy: float = NOMINAL_CY) -> np.ndarray:
    """Flat-ground pinhole projection of ego-frame BEV waypoints to image pixels.

    Args:
        wp_bev: (N, 2) array, col 0 = lateral right-positive m, col 1 = forward m.
        img_h, img_w: original image dimensions (default principal point assumes
            the nominal calibration; we don't rescale cx/cy if image differs).
        cam_height_m: cam mount height above ground.
        cam_forward_offset_m: cam mount forward offset from ego origin.
        fx, fy, cx, cy: pinhole intrinsics.

    Returns:
        (N, 2) pixel (u, v). NaN for points behind / under the camera.
    """
    out = np.full_like(wp_bev, np.nan, dtype=np.float64)
    for i, (lat, fwd) in enumerate(wp_bev):
        # Translate from ego origin to camera origin.
        z_cam = float(fwd) - cam_forward_offset_m
        if z_cam <= 0.5:
            continue
        # Camera frame: x_cam = +right, y_cam = +down.
        x_cam = float(lat)
        y_cam = cam_height_m  # waypoint on the ground; cam is above it -> +down in cam frame
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
        # Allow off-image points; consumer can clip if it wants.
        out[i] = (u, v)
    return out


def blend_seg_masks(img_rgb: np.ndarray, da_mask: np.ndarray, ll_mask: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend drivable area (green) + lane lines (red) onto an RGB image.

    Mirrors the recipe in YOLOPX/lib/utils/plot.show_seg_result (is_demo=True),
    but works on an RGB array and returns a new array.
    """
    out = img_rgb.copy()
    color_seg = np.zeros_like(out, dtype=np.uint8)
    color_seg[da_mask == 1] = (0, 255, 0)   # drivable -> green (RGB)
    color_seg[ll_mask == 1] = (255, 0, 0)   # lane     -> red   (RGB)
    color_mask = color_seg.any(axis=2)
    out[color_mask] = (out[color_mask] * (1 - alpha) +
                      color_seg[color_mask] * alpha).astype(np.uint8)
    return out


def draw_box(img_bgr: np.ndarray, xyxy, label: str, color_bgr=(0, 255, 0),
             line_thickness: int = 2) -> None:
    """Draw a single labelled box (in-place) on a BGR image.

    Like YOLOPX/lib/utils/plot.plot_one_box but with the label drawing actually
    enabled (the upstream version comments it out).
    """
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr,
                  thickness=line_thickness, lineType=cv2.LINE_AA)
    if not label:
        return
    tf = max(line_thickness - 1, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=line_thickness / 3, thickness=tf)
    # Draw label background; clip to top of image if box is at edge.
    y_text_top = max(y1 - th - 4, 0)
    cv2.rectangle(img_bgr, (x1, y_text_top), (x1 + tw + 2, y_text_top + th + 4),
                  color_bgr, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(img_bgr, label, (x1 + 1, y_text_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=line_thickness / 3,
                color=(0, 0, 0), thickness=tf, lineType=cv2.LINE_AA)


def color_for_class(cls_idx: int) -> tuple:
    """BGR colour for one of the 10 BDD classes."""
    return CLASS_COLORS_BGR[int(cls_idx) % len(CLASS_COLORS_BGR)]
