"""Postprocessing for YOLOPX detection + segmentation outputs on Jetson.

Numpy-only port of (lib/core/general.non_max_suppression + scale_coords)
plus the un-pad + argmax block from tools/demo.py:128-144.

Avoids any PyTorch / YOLOPX library import at runtime on Jetson.
"""
from __future__ import annotations

import numpy as np
import cv2


CLASS_NAMES = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor',
               'traffic light', 'traffic sign', 'train']


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    out = np.empty_like(boxes)
    out[:, 0] = x - w / 2
    out[:, 1] = y - h / 2
    out[:, 2] = x + w / 2
    out[:, 3] = y + h / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list:
    """Greedy NMS, numpy. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou < iou_thres]
    return keep


def nms_yolopx(det: np.ndarray, conf_thres: float = 0.30,
               iou_thres: float = 0.45, max_det: int = 300) -> np.ndarray:
    """Numpy port of lib/core/general.non_max_suppression.

    Input:  det shape (num_pred, 5 + nc) — xywh, obj_conf, per-class probs
    Output: (N, 6) — x1, y1, x2, y2, conf, cls   (in 640-space, batch=1)
    """
    if det.ndim == 3:
        det = det[0]
    nc = det.shape[1] - 5

    # Filter by objectness.
    mask = det[:, 4] > conf_thres
    det = det[mask]
    if len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Class scores = obj_conf * cls_conf, pick best.
    det[:, 5:] *= det[:, 4:5]
    boxes = _xywh_to_xyxy(det[:, :4])
    cls_scores = det[:, 5:5 + nc]
    cls = np.argmax(cls_scores, axis=1)
    conf = cls_scores[np.arange(len(cls_scores)), cls]
    mask = conf > conf_thres
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    # Per-class NMS (offset boxes so different classes don't suppress each other).
    offsets = cls.astype(np.float32) * 4096.0
    keep = _nms(boxes + offsets[:, None], conf, iou_thres)
    keep = keep[:max_det]
    return np.concatenate([boxes[keep], conf[keep, None], cls[keep, None]],
                          axis=1).astype(np.float32)


def scale_coords(in_hw: tuple, boxes: np.ndarray, out_hw: tuple) -> np.ndarray:
    """Numpy port of lib/core/general.scale_coords. boxes are xyxy."""
    in_h, in_w = in_hw
    out_h, out_w = out_hw
    gain = min(in_h / out_h, in_w / out_w)
    pad_w = (in_w - out_w * gain) / 2
    pad_h = (in_h - out_h * gain) / 2
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, out_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, out_h)
    return boxes


def segmasks_from_logits(da_logits: np.ndarray, ll_logits: np.ndarray,
                         in_hw: tuple, pad_wh: tuple,
                         out_hw: tuple) -> tuple:
    """Match tools/demo.py:128-144 — un-pad, interpolate to original size,
    argmax, subtract lane from drivable.

    Inputs are (1, 2, H, W) — channel 1 is the foreground class.
    """
    in_h, in_w = in_hw
    pad_w, pad_h = int(pad_wh[0]), int(pad_wh[1])
    out_h, out_w = out_hw

    def _process(logits):
        unpad = logits[:, :, pad_h:in_h - pad_h, pad_w:in_w - pad_w]
        # cv2.resize: per-channel resize requires HWC; we use float32 channel 1 - 0.
        diff = unpad[0, 1] - unpad[0, 0]
        resized = cv2.resize(diff, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return (resized > 0).astype(np.uint8)

    da = _process(da_logits)
    ll = _process(ll_logits)
    da = ((da - ll) == 1).astype(np.uint8)
    return da, ll
