"""Image preprocessing helpers for both engines on Jetson.

Both replicate exactly what the PC PyTorch pipelines did so the TRT engines
see the same numerical inputs they were exported with.
"""
from __future__ import annotations

import numpy as np
import cv2


# ImageNet stats — shared by both models.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _imagenet_normalize(rgb_float: np.ndarray) -> np.ndarray:
    """rgb_float is HxWx3 in [0, 1]. Returns HxWx3 normalized."""
    return (rgb_float - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_dtcp(rgb_uint8: np.ndarray) -> np.ndarray:
    """Match DTCP/deploy/dtcp_infer.py:140-143.

    Stretch (NOT letterbox) to 256x928, ImageNet-normalize, NCHW float32.
    Returns (1, 3, 256, 928) float32 contiguous.
    """
    if rgb_uint8.shape[:2] != (256, 928):
        rgb_uint8 = cv2.resize(rgb_uint8, (928, 256), interpolation=cv2.INTER_LINEAR)
    rgb_f = rgb_uint8.astype(np.float32) / 255.0
    rgb_n = _imagenet_normalize(rgb_f)
    chw = rgb_n.transpose(2, 0, 1)[None]  # (1, 3, 256, 928)
    return np.ascontiguousarray(chw, dtype=np.float32)


def _letterbox(img_bgr: np.ndarray, new_shape=(384, 640), color=(114, 114, 114)):
    """YOLOPX-style letterbox (auto=True, multiple of 32, no scaleFill)."""
    h0, w0 = img_bgr.shape[:2]
    nh_t, nw_t = new_shape
    r = min(nh_t / h0, nw_t / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    dw = (nw_t - nw) / 2
    dh = (nh_t - nh) / 2
    if (h0, w0) != (nh, nw):
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(img_bgr, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (dw, dh)


def preprocess_yolopx(bgr_uint8: np.ndarray, in_shape=(384, 640)) -> tuple:
    """Match YOLOPX/lib/dataset/DemoDataset + tools/demo.py preprocessing.

    Returns:
        x        — (1, 3, H, W) float32, contiguous, ImageNet-normalized
        h0, w0   — original image size (for box rescaling)
        pad_wh   — (dw, dh) padding applied (for mask un-pad)
        ratio    — letterbox scale (for box rescaling)
    """
    padded_bgr, ratio, (dw, dh) = _letterbox(bgr_uint8, new_shape=in_shape)
    rgb = padded_bgr[..., ::-1].copy()
    rgb_f = rgb.astype(np.float32) / 255.0
    rgb_n = _imagenet_normalize(rgb_f)
    chw = rgb_n.transpose(2, 0, 1)[None]
    return (np.ascontiguousarray(chw, dtype=np.float32),
            bgr_uint8.shape[0], bgr_uint8.shape[1],
            (dw, dh), ratio)
