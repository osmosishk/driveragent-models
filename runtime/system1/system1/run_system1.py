"""
System1 inference (Path A: pure PyTorch, no TRT).

Usage:
  # Smoke test + latency benchmark with dummy input
  python deploy/run_system1.py --checkpoint deploy/system1_deploy.pth --benchmark

  # Single sample inference
  python deploy/run_system1.py --checkpoint deploy/system1_deploy.pth --input sample.pt

This is the script to run on both the 4090 (sanity check) and the Jetson AGX Orin
(target deployment). It loads the self-contained system1_deploy.pth, reconstructs
config + vocabulary, builds the model, and runs forward.
"""

import argparse
import os
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from system1.config import System1Config
from system1.system1_model import System1Model


def load_model(checkpoint_path, device='cuda'):
    pkg = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    cfg_dict = pkg['config']
    config = System1Config()
    valid_keys = {f.name for f in fields(System1Config)}
    for k, v in cfg_dict.items():
        if k in valid_keys:
            setattr(config, k, v)

    tmp_dir = tempfile.mkdtemp(prefix='system1_vocab_')
    vocab = pkg['vocabulary']
    path_file = os.path.join(tmp_dir, 'path.npy')
    vel_file = os.path.join(tmp_dir, 'vel.npy')
    traj_file = os.path.join(tmp_dir, 'traj.npz')
    np.save(path_file, vocab['path_anchors'])
    np.save(vel_file, vocab['vel_anchors'])
    np.savez(traj_file,
             trajectory=vocab['traj_trajectory'],
             trajectory_mask=vocab['traj_trajectory_mask'])
    config.path_anchor_file = path_file
    config.velocity_anchor_file = vel_file
    config.trajectory_anchor_file = traj_file

    model = System1Model(config)
    missing, unexpected = model.load_state_dict(pkg['model_state_dict'], strict=False)
    if missing:
        print(f"  warning: {len(missing)} missing keys in state_dict")
    if unexpected:
        print(f"  warning: {len(unexpected)} unexpected keys in state_dict")
    model = model.to(device).eval()

    info = pkg['model_info']
    print(f"Loaded System1 {info['version']} (epoch {info['epoch']})")
    print(f"  Backbone: {info['backbone']}")
    print(f"  Heading head: {info['heading_head']}, gate sigma: {info['gate_sigma']}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    return model, pkg['preprocessing']


def preprocess_images(raw_images, preproc_cfg, device='cuda'):
    from torchvision.transforms.functional import resize as tv_resize

    if raw_images.dtype == torch.uint8:
        raw_images = raw_images.float() / 255.0

    H_raw, W_raw = preproc_cfg['raw_size']
    H_final, W_final = preproc_cfg['final_size']
    scale = preproc_cfg['resize']

    new_h = int(H_raw * scale)
    new_w = int(W_raw * scale)

    resized = [tv_resize(raw_images[c], [new_h, new_w], antialias=True) for c in range(6)]
    images = torch.stack(resized)

    crop_y = 0
    crop_x = max(0, (new_w - W_final) // 2)
    images = images[:, :, crop_y:crop_y + H_final, crop_x:crop_x + W_final]

    mean = torch.tensor(preproc_cfg['mean']).view(1, 3, 1, 1)
    std = torch.tensor(preproc_cfg['std']).view(1, 3, 1, 1)
    images = (images - mean) / std

    images = images.unsqueeze(0).to(device)
    return images


_AUTOCAST_DTYPES = {
    'fp32': None,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


@torch.no_grad()
def run_inference(model, images, ego_state, projection_mat, image_wh,
                  scene_ctx=None, device='cuda', precision='bf16'):
    camera_metas = {
        'projection_mat': projection_mat.to(device),
        'image_wh': image_wh.to(device),
    }

    is_cuda = device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')
    autocast_dtype = _AUTOCAST_DTYPES[precision]

    if is_cuda and autocast_dtype is not None:
        ctx = torch.amp.autocast('cuda', dtype=autocast_dtype)
    else:
        ctx = torch.amp.autocast('cuda', enabled=False) if is_cuda \
            else torch.amp.autocast('cpu', enabled=False)

    with ctx:
        output, _ = model(
            images.to(device),
            ego_state.to(device),
            camera_metas,
            targets=None,
            scene_ctx=scene_ctx,
        )

    trajectory = output['trajectory'][0].float().cpu()
    return trajectory, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='deploy/system1_deploy.pth')
    parser.add_argument('--input', default=None, help='Path to sample .pt file')
    parser.add_argument('--benchmark', action='store_true', help='Run latency benchmark')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--warmup', type=int, default=5, help='Benchmark warmup iters')
    parser.add_argument('--precision', choices=list(_AUTOCAST_DTYPES.keys()),
                        default='bf16',
                        help='Autocast precision. fp16 produces NaN on system1_v9e (heading_head); '
                             'bf16 matches fp32 to ~0.01 and is recommended.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model, preproc = load_model(args.checkpoint, device)

    if args.benchmark:
        print(f"\n=== Latency Benchmark (precision={args.precision}, dummy input) ===")
        # Zeros = mean-gray after normalization. Random noise can overflow
        # the backbone in FP16 and produce NaN downstream, which masks real
        # latency/correctness signal.
        dummy_images = torch.zeros(1, 6, 3, 256, 704, device=device)
        dummy_ego = torch.zeros(1, 8, device=device)
        # Plausible pinhole projection (704x256 image, fx=fy=700) so that
        # 3D keypoints project to valid image coords and grid_sample returns finite values.
        # See DEPLOY_INSTRUCTIONS.md: "NaN in output / Check that projection_mat is valid".
        K = torch.tensor([
            [700.0,   0.0, 352.0, 0.0],
            [  0.0, 700.0, 128.0, 0.0],
            [  0.0,   0.0,   1.0, 0.0],
            [  0.0,   0.0,   0.0, 1.0],
        ])
        dummy_proj = K.unsqueeze(0).unsqueeze(0).expand(1, 6, 4, 4).contiguous().to(device)
        dummy_wh = torch.tensor([[[704, 256]]], dtype=torch.float32).expand(1, 6, 2).contiguous().to(device)

        for _ in range(args.warmup):
            run_inference(model, dummy_images, dummy_ego, dummy_proj, dummy_wh, device=device, precision=args.precision)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(args.iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            traj, _ = run_inference(model, dummy_images, dummy_ego, dummy_proj, dummy_wh, device=device, precision=args.precision)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        print(f"  Latency: {times.mean():.1f} +/- {times.std():.1f} ms  (n={args.iters})")
        print(f"  Min: {times.min():.1f} ms, Max: {times.max():.1f} ms, P50: {np.median(times):.1f} ms")
        print(f"  FPS: {1000 / times.mean():.1f}")
        print(f"  Trajectory output shape: {tuple(traj.shape)}")
        finite = torch.isfinite(traj).all().item()
        print(f"  Output finite: {finite}")
        return

    if args.input:
        sample = torch.load(args.input, map_location='cpu', weights_only=False)
        raw_images = sample['img']
        images = preprocess_images(raw_images, preproc, device)

        ego_state = sample.get('ego_state', torch.zeros(1, 8))
        if ego_state.dim() == 1:
            ego_state = ego_state.unsqueeze(0)

        projection_mat = sample['projection_mat'].unsqueeze(0)
        image_wh = torch.tensor([[[704, 256]]], dtype=torch.float32).expand(1, 6, 2)

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

        t0 = time.perf_counter()
        trajectory, _ = run_inference(model, images, ego_state, projection_mat, image_wh,
                                      scene_ctx=scene_ctx, device=device, precision=args.precision)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\nTrajectory ({elapsed:.1f} ms):")
        print(f"  {'t(s)':>5s}  {'x(m)':>8s}  {'y(m)':>8s}  {'heading(deg)':>12s}")
        for t in range(trajectory.shape[0]):
            x, y, h = trajectory[t].tolist()
            print(f"  {(t+1)*0.5:5.1f}  {x:8.3f}  {y:8.3f}  {np.degrees(h):12.1f}")
    else:
        print("\nModel loaded successfully. Use --benchmark or --input <sample.pt>")


if __name__ == '__main__':
    main()
