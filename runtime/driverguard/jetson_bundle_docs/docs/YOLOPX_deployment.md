# YOLOPX v2 — Training Summary & Deployment Guide

## 1. What was done

Original `weights/epoch-195.pth` was trained `single_cls=True` (vehicles only, 1 class). For ADAS use we needed all 10 BDD100K detection classes. A first 60-epoch retrain (v1) plateaued at mAP@0.5 = 0.063 — investigation found two latent bugs in the upstream YOLOPX loss code that were harmless under `single_cls=True` but fatal for multi-class. After fixing them, a 30-epoch retrain (v2) reached mAP@0.5 = 0.537.

### Bugs fixed

| File | Issue | Fix |
|---|---|---|
| `lib/models/YOLOX_Loss.py:254` | Classification BCE loss was hard-coded to `0.0` (block commented out). Total loss = `5×iou + obj` only — class predictions got zero gradient | Re-enabled the `loss_cls` block; total = `5×iou + obj + cls` |
| `lib/core/loss.py:93` | Detection loss multiplied by `0.02` while seg losses used `0.2` (10× under-weighted) | Raised det multiplier to `0.2` (parity with seg) |
| `lib/core/function.py:108-112` | Only logged total loss; per-head components invisible | Added per-head TensorBoard scalars + extended train log line |

### Multi-class adaptation

| File | Change |
|---|---|
| `lib/dataset/convert.py` | `id_dict` rewritten to 10 stock BDD100K classes (`person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train`) |
| `lib/dataset/bdd.py` | `single_cls = False`; removed traffic-light-color split |
| `lib/models/YOLOP.py` | YOLOXHead `nc=10`; `self.names` populated with class strings |
| `lib/core/loss.py` | `YOLOX_Loss(device, 10)` |
| `tools/train.py` | `model.nc = 10`; new partial-load logic with shape filter (drops only `cls_preds` when nc changes) |
| `tools/test.py` | `model.nc = 10` |
| `tools/demo.py` | Box label includes class name + per-class color (was conf-only, single yellow) |

### Training results (v2)

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Recall | DA mIOU | LL IOU |
|---|---|---|---|---|---|
| 2  | 0.396 | 0.195 | 0.676 | 0.925 | 0.208 |
| 10 | 0.479 | 0.243 | 0.700 | 0.927 | 0.203 |
| 18 | 0.509 | 0.263 | 0.732 | 0.927 | 0.213 |
| 24 | 0.527 | 0.278 | 0.742 | 0.929 | 0.205 |
| **30** | **0.537** | **0.285** | **0.749** | **0.929** | **0.207** |

Total wall time: 8h 35min on a single RTX 4090. Warmstarted from v1's `epoch-50.pth` (which had usable backbone + seg heads despite broken detection).

---

## 2. Model artifact

| Property | Value |
|---|---|
| **Final checkpoint** | `runs/BddDataset/_2026-05-10-08-32/epoch-30.pth` |
| Format | PyTorch state-dict wrapped as `{'epoch', 'state_dict', 'best_state_dict', ...}` |
| Architecture | ELANNet backbone → PaFPNELAN neck → 3 heads (YOLOX detection, DA seg, LL seg) |
| Parameters | ~30M |
| Input | RGB image, **640×640**, normalized with ImageNet mean/std (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`), letterbox-padded |
| Outputs | `(det_out, da_seg_out, ll_seg_out)` — det predictions need NMS; seg outputs are 2-channel logits at 640×640 |
| Classes | `0:person, 1:rider, 2:car, 3:bus, 4:truck, 5:bike, 6:motor, 7:traffic light, 8:traffic sign, 9:train` |

There is also a `final_state.pth` in the same folder — it's saved as a **bare** state_dict (no wrapper), so `demo.py`/`test.py` (which expect `checkpoint['state_dict']`) won't load it. Always use `epoch-30.pth`.

---

## 3. Local PyTorch inference (validation on PC)

```bash
cd ~/development/YOLOPX
python tools/demo.py \
    --weights runs/BddDataset/_2026-05-10-08-32/epoch-30.pth \
    --source <path_to_images_or_video> \
    --conf-thres 0.30 \
    --iou-thres 0.45 \
    --save-dir inference/output
```

- `--source` accepts a folder of images, a single video file, or a webcam index (`0`)
- `--conf-thres 0.30` is a sane default; 0.25 keeps more recall, 0.35 cleans false positives
- Latency on RTX 4090: ~3.5 ms inference + ~1.5 ms NMS per 640×640 frame
- Per-class evaluation: `python tools/test.py --weights runs/BddDataset/_2026-05-10-08-32/epoch-30.pth` (prints AP per class)

---

## 4. Jetson deployment — Option A (PyTorch, easiest)

Use this for the **first deployment / debugging**. Confirms behavior matches PC before TRT conversion.

### Prerequisites
- Jetson Orin (Nano / NX / AGX) with JetPack 5.1+
- NVIDIA's PyTorch wheel matching JetPack version (https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- ~3 GB disk for PyTorch + dependencies

### Steps
```bash
# On Jetson:
pip install --user torch torchvision  # use NVIDIA's wheel URL for your JetPack
pip install --user -r requirements.txt
scp -r <pc>:~/development/YOLOPX ~/YOLOPX
scp <pc>:~/development/YOLOPX/runs/BddDataset/_2026-05-10-08-32/epoch-30.pth ~/YOLOPX/weights/

# Run inference:
cd ~/YOLOPX
python tools/demo.py \
    --weights weights/epoch-30.pth \
    --source <image_folder_or_video> \
    --conf-thres 0.30 \
    --device 0 \
    --save-dir inference/output
```

### Expected performance (FP16, 640×640)
- Orin Nano (8 GB): **~10–20 FPS**
- Orin NX: **~25–35 FPS**
- Orin AGX: **~40–60 FPS**

Half-precision is enabled automatically (see `tools/demo.py:48,57`).

---

## 5. Jetson deployment — Option B (TensorRT, recommended for trial / production)

3–5× faster than Option A, no PyTorch needed at runtime, smaller memory footprint.

### 5.1 Export PyTorch → ONNX (run on PC, has more disk/RAM)

Save the script below as `tools/export_onnx.py`:

```python
import argparse, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from lib.config import cfg
from lib.models import get_net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="yolopx_v2.onnx")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    model = get_net(cfg)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.zeros(1, 3, args.img_size, args.img_size)

    # YOLOPX returns (det_out, da_seg_out, ll_seg_out) where det_out is a tuple
    # (inf_out, train_out). For inference we only need inf_out, da_seg_out, ll_seg_out.
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            det_out, da, ll = self.m(x)
            inf_out, _ = det_out
            return inf_out, da, ll

    wrapper = InferenceWrapper(model).eval()

    torch.onnx.export(
        wrapper, dummy, args.out,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["det", "da_seg", "ll_seg"],
        dynamic_axes={"image": {0: "batch"}, "det": {0: "batch"},
                      "da_seg": {0: "batch"}, "ll_seg": {0: "batch"}},
    )
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
```

Run:
```bash
python tools/export_onnx.py \
    --weights runs/BddDataset/_2026-05-10-08-32/epoch-30.pth \
    --out yolopx_v2.onnx
```

Validate the ONNX:
```bash
pip install onnxruntime onnx
python -c "import onnx; m=onnx.load('yolopx_v2.onnx'); onnx.checker.check_model(m); print('OK')"
```

### 5.2 Convert ONNX → TensorRT engine (run **on the Jetson**, since engines are device-specific)

```bash
# Copy yolopx_v2.onnx to Jetson, then:
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolopx_v2.onnx \
    --saveEngine=yolopx_v2_fp16.engine \
    --fp16 \
    --workspace=4096 \
    --verbose
```

For INT8 (another ~1.5× speedup, but requires calibration):
```bash
# Provide a calibration folder of ~500 representative images first
trtexec --onnx=yolopx_v2.onnx --saveEngine=yolopx_v2_int8.engine \
    --int8 --calib=<calib.cache> --workspace=4096
```

### 5.3 Runtime inference (Python, TensorRT)

Skeleton (`infer_trt.py` on Jetson):
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

CLASS_NAMES = ['person','rider','car','bus','truck','bike','motor',
               'traffic light','traffic sign','train']

def load_engine(path):
    with open(path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as r:
        return r.deserialize_cuda_engine(f.read())

def preprocess(img, size=640):
    # Letterbox to square, normalize ImageNet stats (matches training pipeline)
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(h*s), int(w*s)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size-nh)//2, (size-nw)//2
    canvas[top:top+nh, left:left+nw] = resized
    x = canvas[..., ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
    x = (x - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
    x = x.transpose(2,0,1)[None]  # NCHW
    return np.ascontiguousarray(x.astype(np.float16))  # FP16 engine

# Engine load → allocate buffers → cuda.memcpy → context.execute_v2 → postprocess (NMS)
# Reuse non_max_suppression from lib/core/general.py (numpy port) for det postprocess.
```

For NMS on Jetson, port `non_max_suppression` from `lib/core/general.py` to numpy or use TensorRT's built-in `EfficientNMS_TRT` plugin (better — runs on GPU). Adding `EfficientNMS_TRT` requires modifying the ONNX graph; tools like `onnx-graphsurgeon` automate this.

### Expected performance (TensorRT FP16, 640×640)
- Orin Nano: **~50–80 FPS**
- Orin NX: **~80–120 FPS**
- Orin AGX: **~150–250 FPS**

---

## 6. Pre-deployment checklist

Before running **on a vehicle / robot** in public, especially for the Cambridge low-speed micromobility trial:

1. **Per-class AP validation.** Run `python tools/test.py --weights runs/BddDataset/_2026-05-10-08-32/epoch-30.pth`. If `person` mAP@0.5 < 0.30, retrain with class-balanced cls BCE before any public trial.
2. **In-domain validation.** Collect ~500 frames from the **actual deployment camera** at the trial site. Hand-label or visually inspect inference output for failure modes (low sun, rain, motion blur).
3. **Latency budget.** Measure end-to-end latency from camera capture → bounding boxes on the actual Jetson (not just inference time). Camera I/O + preprocess + postprocess often dominates.
4. **Defensive layer.** Don't rely on a single model for safety-critical stops. Add a fallback: depth-based proximity stop, or a parallel pedestrian-only detector (YOLOX-S COCO-`person` has higher recall).
5. **Image-size mismatch.** Model is fixed at 640×640. If your camera is 1920×1080, replicate the letterbox preprocess from `lib/dataset/AutoDriveDataset.py`. Don't naively resize — aspect ratio matters.
6. **Class label visibility.** `tools/demo.py` already shows class name + confidence (fixed during nuScenes test). For your runtime wrapper, hardcode the 10 names rather than reading from the .pth — keeps inference free of PyTorch dependency.

---

## 7. Quick reference

| Task | Command |
|---|---|
| Full retrain | `python tools/train.py` |
| Per-class evaluation | `python tools/test.py --weights <ckpt>` |
| Demo / inference | `python tools/demo.py --weights <ckpt> --source <path> --save-dir <out>` |
| TensorBoard | `tensorboard --logdir runs/` |
| ONNX export | `python tools/export_onnx.py --weights <ckpt> --out yolopx_v2.onnx` *(after creating the script in §5.1)* |
| TRT engine build | `trtexec --onnx=yolopx_v2.onnx --saveEngine=engine.trt --fp16` *(on Jetson)* |

| Path | Contents |
|---|---|
| `runs/BddDataset/_2026-05-10-08-32/` | v2 training run, all checkpoints + tfevents |
| `runs/BddDataset/_2026-05-10-08-32/epoch-30.pth` | **Final deployable model** |
| `runs/BddDataset/_2026-05-09-17-59/` | v1 training run (broken cls; kept for archaeology) |
| `weights/epoch-195.pth` | Original upstream checkpoint (1-class) |
| `inference/nuscenes_output/` | Sample inference on 200 nuScenes frames (for visual sanity check) |
