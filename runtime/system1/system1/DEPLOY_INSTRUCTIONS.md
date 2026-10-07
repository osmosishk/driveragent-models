# System1 Deployment on Jetson AGX Orin

## Overview

Deploy the System1 autonomous driving model (ConvNeXt V2 Tiny + Factorized Trajectory Scorer) for real-time inference on Jetson AGX Orin.

**Model**: System1 v9e (epoch 12)
- Detection backbone: ConvNeXt V2 Tiny + FPN (30M params)
- Trajectory scorer: 2-layer factorized decoder with heading head (17M params)
- Total: 47.6M params
- Input: 6 cameras × 3 × 256 × 704
- Output: trajectory [B, 6, 3] (x, y, heading) at 0.5s intervals (3s horizon)

**Performance targets (paper)**: e2e ADE 0.831m, heading error 50.7°

---

## Step 1: Environment Setup

### Prerequisites
- JetPack 6.x (L4T 36.x) with PyTorch 2.x and CUDA 12.x
- Python 3.10+

### Install dependencies

```bash
# Core
pip install torch torchvision  # Use NVIDIA's Jetson wheel
pip install timm numpy

# For the deformable attention CUDA kernel (optional, has fallback)
# Build from source if needed:
cd /path/to/sparsedrive/system1/ops
python setup.py install
# If build fails, the model uses F.grid_sample() fallback automatically
```

---

## Step 2: Transfer Files

Copy these from the training server to the Jetson:

```bash
# Required
deploy/system1_deploy.pth         # 252 MB — full model weights + vocab + config

# Required code (the full system1 module + backbone)
system1/                          # System1 model code
models_convnext/                  # ConvNeXt backbone code
models/backbone.py                # FPN + GridMask (imported by models_convnext)
models/fpn.py                     # FPN implementation

# Optional (for hybrid TRT+PyTorch later)
deploy/backbone_nchw.onnx         # ONNX backbone (experimental)
deploy/backbone_nchw.onnx.data    # ONNX weights
deploy/system1_scorer.pth         # Scorer-only package
```

### Directory structure on Jetson

```
/path/to/sparsedrive/
├── system1/                      # Model code
│   ├── __init__.py
│   ├── config.py
│   ├── system1_model.py
│   ├── scorer/
│   │   ├── __init__.py
│   │   ├── scorer_head.py
│   │   ├── decoder_layer.py
│   │   ├── deformable_agg.py
│   │   ├── embedders.py
│   │   ├── keypoints_generator.py
│   │   ├── losses.py
│   │   └── vocabulary.py
│   └── ops/                      # Optional CUDA kernels
├── models_convnext/
│   └── backbone.py
├── models/
│   ├── backbone.py               # FPN, GridMask
│   └── fpn.py
└── deploy/
    └── system1_deploy.pth
```

---

## Step 3: Load and Run Inference

### Basic inference script

```python
"""
System1 inference on Jetson AGX Orin.

Usage:
  python run_system1.py --checkpoint deploy/system1_deploy.pth --input sample.pt
"""

import sys
import time
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from system1.config import System1Config
from system1.system1_model import System1Model


def load_model(checkpoint_path, device='cuda'):
    """Load System1 from the deploy checkpoint."""
    pkg = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Reconstruct config from saved dict
    cfg_dict = pkg['config']
    config = System1Config()

    # Write vocabulary to temp files (config expects file paths)
    import tempfile, os
    tmp_dir = tempfile.mkdtemp()
    vocab = pkg['vocabulary']

    path_file = os.path.join(tmp_dir, 'path.npy')
    vel_file = os.path.join(tmp_dir, 'vel.npy')
    traj_file = os.path.join(tmp_dir, 'traj.npz')

    np.save(path_file, vocab['path_anchors'])
    np.save(vel_file, vocab['vel_anchors'])
    np.savez(traj_file, trajectory=vocab['traj_trajectory'],
             trajectory_mask=vocab['traj_trajectory_mask'])

    config.path_anchor_file = path_file
    config.velocity_anchor_file = vel_file
    config.trajectory_anchor_file = traj_file

    # Build model
    model = System1Model(config)
    model.load_state_dict(pkg['model_state_dict'], strict=False)
    model = model.to(device).eval()

    print(f"Loaded System1 v{pkg['model_info']['version']} "
          f"(epoch {pkg['model_info']['epoch']})")
    print(f"  Backbone: {pkg['model_info']['backbone']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    return model, pkg['preprocessing']


def preprocess_images(raw_images, preproc_cfg, device='cuda'):
    """
    Preprocess raw camera images for System1.

    Args:
        raw_images: [6, 3, 900, 1600] raw camera images (uint8 or float [0,1])
        preproc_cfg: dict with mean, std, resize, crop, final_size, raw_size

    Returns:
        images: [1, 6, 3, 256, 704] normalized tensor
        projection_mat: needs to be updated for resize+crop (pass through separately)
    """
    from torchvision.transforms.functional import resize as tv_resize

    if raw_images.dtype == torch.uint8:
        raw_images = raw_images.float() / 255.0

    H_raw, W_raw = preproc_cfg['raw_size']
    H_final, W_final = preproc_cfg['final_size']
    scale = preproc_cfg['resize']  # 0.44

    new_h = int(H_raw * scale)   # 396
    new_w = int(W_raw * scale)   # 704

    # Resize each camera
    resized = []
    for c in range(6):
        r = tv_resize(raw_images[c], [new_h, new_w], antialias=True)
        resized.append(r)
    images = torch.stack(resized)  # [6, 3, 396, 704]

    # Top-crop to 256x704
    crop_y = 0
    crop_x = max(0, (new_w - W_final) // 2)
    images = images[:, :, crop_y:crop_y+H_final, crop_x:crop_x+W_final]

    # Add batch dim
    images = images.unsqueeze(0).to(device)  # [1, 6, 3, 256, 704]

    # Note: projection matrices must also be scaled and cropped:
    #   proj[:, 0] *= scale    (x scaling)
    #   proj[:, 1] *= scale    (y scaling)
    #   proj[:, 0, 2] -= crop_x
    #   proj[:, 1, 2] -= crop_y

    return images


@torch.no_grad()
def run_inference(model, images, ego_state, projection_mat, image_wh,
                  scene_ctx=None, device='cuda'):
    """
    Run System1 inference.

    Args:
        model: System1Model
        images: [1, 6, 3, 256, 704] preprocessed camera images
        ego_state: [1, 8] ego vehicle state
            [vx, vy, ax, ay, yaw_rate, speed, cmd_onehot_0, cmd_onehot_1]
            cmd_onehot: [1,0]=left, [0,0]=straight, [0,1]=right
        projection_mat: [1, 6, 4, 4] lidar-to-image projection matrices
        image_wh: [1, 6, 2] image width/height after preprocessing
        scene_ctx: optional dict with detected agents + map context
            agent_ctx: [1, N, 9] — [x,y,z,l,w,h,yaw,vx,vy] per agent
            agent_mask: [1, N] — True for valid agents
            map_ctx: [1, M, 40] — 20 polyline points × 2D
            map_mask: [1, M] — True for valid map elements

    Returns:
        trajectory: [6, 3] — 6 waypoints × (x, y, heading) in ego frame
            x,y in meters relative to ego position
            heading in radians
    """
    camera_metas = {
        'projection_mat': projection_mat.to(device),
        'image_wh': image_wh.to(device),
    }

    with torch.amp.autocast('cuda', dtype=torch.float16):
        output, _ = model(
            images.to(device),
            ego_state.to(device),
            camera_metas,
            targets=None,
            scene_ctx=scene_ctx,
        )

    trajectory = output['trajectory'][0].cpu()  # [6, 3]
    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='deploy/system1_deploy.pth')
    parser.add_argument('--input', default=None, help='Path to sample .pt file')
    parser.add_argument('--benchmark', action='store_true', help='Run latency benchmark')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model, preproc = load_model(args.checkpoint, device)

    if args.benchmark:
        print("\n=== Latency Benchmark ===")
        dummy_images = torch.randn(1, 6, 3, 256, 704, device=device)
        dummy_ego = torch.zeros(1, 8, device=device)
        dummy_proj = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 6, 4, 4).to(device)
        dummy_wh = torch.tensor([[[704, 256]]], dtype=torch.float32).expand(1, 6, 2).to(device)

        # Warmup
        for _ in range(5):
            run_inference(model, dummy_images, dummy_ego, dummy_proj, dummy_wh, device=device)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            traj = run_inference(model, dummy_images, dummy_ego, dummy_proj, dummy_wh, device=device)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"  Latency: {times.mean():.1f} ± {times.std():.1f} ms")
        print(f"  Min: {times.min():.1f} ms, Max: {times.max():.1f} ms")
        print(f"  FPS: {1000 / times.mean():.1f}")
        print(f"  Trajectory output shape: {traj.shape}")
        return

    if args.input:
        sample = torch.load(args.input, map_location='cpu', weights_only=False)

        # Preprocess
        raw_images = sample['img']  # [6, 3, H, W]
        images = preprocess_images(raw_images, preproc, device)

        # Ego state (zeros if not available)
        ego_state = sample.get('ego_state', torch.zeros(1, 8))
        if ego_state.dim() == 1:
            ego_state = ego_state.unsqueeze(0)

        # Projection matrices
        projection_mat = sample['projection_mat'].unsqueeze(0)  # [1, 6, 4, 4]
        image_wh = torch.tensor([[[704, 256]]], dtype=torch.float32).expand(1, 6, 2)

        # Scene context (if available)
        scene_ctx = None
        if 'gt_boxes' in sample:
            N = sample['gt_boxes'].shape[0]
            agent_ctx = torch.zeros(1, max(N, 1), 9)
            agent_mask = torch.zeros(1, max(N, 1), dtype=torch.bool)
            if N > 0:
                agent_ctx[0, :N] = sample['gt_boxes'][:, :9]
                agent_mask[0, :N] = True
            scene_ctx = {
                'agent_ctx': agent_ctx.to(device),
                'agent_mask': agent_mask.to(device),
                'map_ctx': torch.zeros(1, 1, 40, device=device),
                'map_mask': torch.zeros(1, 1, dtype=torch.bool, device=device),
            }

        # Run inference
        t0 = time.perf_counter()
        trajectory = run_inference(model, images, ego_state,
                                   projection_mat, image_wh,
                                   scene_ctx=scene_ctx, device=device)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\nTrajectory ({elapsed:.1f} ms):")
        print(f"  {'t(s)':>5s}  {'x(m)':>8s}  {'y(m)':>8s}  {'heading(°)':>10s}")
        for t in range(6):
            x, y, h = trajectory[t].tolist()
            print(f"  {(t+1)*0.5:5.1f}  {x:8.3f}  {y:8.3f}  {np.degrees(h):10.1f}")
    else:
        print("\nModel loaded successfully. Use --input <sample.pt> or --benchmark")


if __name__ == '__main__':
    main()
```

---

## Step 4: Integration with Camera Pipeline

### Input format

```python
# Raw camera images from 6 cameras
raw_images: torch.Tensor  # [6, 3, 900, 1600], uint8 or float [0,1]

# Camera calibration (lidar-to-image 4×4 projection matrices)
projection_mat: torch.Tensor  # [6, 4, 4], float32

# Ego vehicle state
ego_state: torch.Tensor  # [8]
# [vx, vy, ax, ay, yaw_rate, speed, cmd_onehot_0, cmd_onehot_1]
# cmd_onehot: [1,0]=turn left, [0,0]=go straight, [0,1]=turn right

# Scene context from perception (optional but improves accuracy)
agent_ctx: torch.Tensor   # [N, 9] — [x,y,z,l,w,h,yaw,vx,vy] per detected agent
agent_mask: torch.Tensor  # [N] — True for valid agents
map_ctx: torch.Tensor     # [M, 40] — 20 polyline points × (x,y) per map element
map_mask: torch.Tensor    # [M] — True for valid map elements
```

### Output format

```python
trajectory: torch.Tensor  # [6, 3] — 6 future waypoints in ego frame
# Each waypoint: [x_meters, y_meters, heading_radians]
# Timesteps: 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s
# x: forward (positive = ahead)
# y: lateral (positive = left)
# heading: radians, 0 = forward
```

### Preprocessing notes

The model expects images preprocessed as follows:
1. **Resize**: scale by 0.44 (1600×900 → 704×396)
2. **Crop**: top-crop to 256×704 (rows 0:256)
3. **Normalize**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Projection update**: scale projection matrix rows 0,1 by 0.44, subtract crop offsets from translation

If your cameras have different resolution than 1600×900, adjust the resize scale to produce at least 704×256 after resize, then crop.

---

## Step 5: Latency Benchmark

```bash
python run_system1.py --checkpoint deploy/system1_deploy.pth --benchmark
```

### Expected performance on Jetson AGX Orin 64GB

| Mode | Estimated Latency | Notes |
|------|------------------|-------|
| FP32 | ~150-250ms | PyTorch default |
| FP16 | ~60-120ms | `torch.amp.autocast` (used in script) |
| TRT backbone + FP16 scorer | ~30-50ms | Requires ONNX fix (see below) |

---

## Step 6: TRT Backbone Acceleration (Optional, Later)

The ONNX backbone (`backbone_nchw.onnx`) uses GroupNorm substitution for LayerNorm to avoid TRT Transpose fusion failures. This substitution is NOT numerically exact (cosine similarity ~0.95-0.97).

### Known TRT issue on Orin

TRT 10.3 on sm_87 cannot compile the ConvNeXt ONNX pattern:
```
Conv (NCHW) → Transpose [0,2,3,1] → LayerNorm (NHWC) → Transpose [0,3,1,2] → Conv
```

### Correct fix (TODO)

Instead of GroupNorm substitution, decompose LayerNorm into element-wise NCHW ops:
```python
# LayerNorm(C) on NHWC = normalize last dim
# Equivalent NCHW ops (no Transpose):
mean = x.mean(dim=1, keepdim=True)         # [N,1,H,W]
var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
x_norm = (x - mean) / (var + eps).sqrt()
x_out = weight.view(1,C,1,1) * x_norm + bias.view(1,C,1,1)
```

This requires patching the ConvNeXt block forward to:
1. Remove permute calls
2. Replace LayerNorm with the above ops
3. Convert Linear MLP to Conv1x1

After patching, re-export and build on Orin:
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=backbone_nchw_fixed.onnx \
    --saveEngine=backbone_fp16.trt \
    --fp16 --memPoolSize=workspace:4096MiB
```

---

## Troubleshooting

### "No module named 'system1'"
Ensure the project root is in `sys.path` or `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/sparsedrive:$PYTHONPATH
```

### "No module named 'timm'"
```bash
pip install timm
```

### Deformable attention CUDA kernel not found
The model falls back to `F.grid_sample()` automatically. No action needed. To build the CUDA kernel for ~15% speedup:
```bash
cd system1/ops && python setup.py install
```

### Out of GPU memory
Reduce to FP16 (already enabled via autocast in the script). If still OOM, process cameras in two batches of 3 instead of all 6.

### NaN in output
Check that `projection_mat` is valid (not all zeros). The deformable attention projects 3D keypoints to 2D using these matrices.

---

## Model Architecture Reference

```
System1Model (47.6M params)
├── backbone: SparseDriveConvNeXtBackbone (30M)
│   ├── convnext: ConvNeXt V2 Tiny (28M)
│   │   └── 4 stages: [96, 192, 384, 768] channels
│   └── fpn: FPN → 4 levels × 256 channels
│
└── scorer: FactorizedScorer (17M)
    ├── vocabulary: 1024 paths × 256 velocities (frozen)
    ├── path_embedder: PathEmbedder
    ├── vel_embedder: VelocityEmbedder
    ├── status_encoder: Linear(8 → 256)
    └── decoder: 2 × ScorerDecoderLayer
        ├── Layer 0 (coarse): 1024→128 paths, 256→64 velocities
        └── Layer 1 (fine): 128→20 paths, 64→20 velocities
            ├── traj scoring: 20×20=400 trajectories → best 1
            ├── 8 metric heads (collision, comfort, etc.)
            └── heading_head: MLP(256→1024→6) → per-timestep heading

Scene context (gated cross-attention, σ=0.047):
  ├── agent_encoder: Linear(9→256) — [x,y,z,l,w,h,yaw,vx,vy]
  ├── map_encoder: Linear(40→256) — 20 polyline points × 2D
  ├── scene_gate: sigmoid-gated pooled bias
  └── path_agent/map attention: cross-attention with LN-gated residual
```
