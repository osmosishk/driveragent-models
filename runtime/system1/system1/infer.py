"""
System1 Inference on Jetson AGX Orin.

Usage:
  # PyTorch-only (no TRT needed):
  python infer.py --images /path/to/6_cameras.pt

  # With TRT backbone:
  python infer.py --images /path/to/6_cameras.pt --trt_engine backbone_fp16.trt
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


def load_scorer(package_path, device='cuda'):
    """Load scorer from exported package."""
    pkg = torch.load(package_path, map_location='cpu', weights_only=False)

    config = pkg['config']
    vocab = pkg['vocabulary']
    scorer_sd = pkg['scorer_state_dict']

    print(f"Model: {pkg['model_info']['version']} (epoch {pkg['model_info']['epoch']})")
    print(f"  Heading head: {pkg['model_info']['heading_head']}")

    return config, vocab, scorer_sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scorer_pkg', default='system1_scorer.pth')
    p.add_argument('--backbone_onnx', default='backbone_nchw.onnx')
    p.add_argument('--trt_engine', default=None, help='TRT engine (build from ONNX)')
    p.add_argument('--images', required=True, help='Path to input tensor [6, 3, 256, 704]')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load scorer
    config, vocab, scorer_sd = load_scorer(args.scorer_pkg, device)
    print(f"Scorer loaded: {len(scorer_sd)} keys")
    print(f"Vocabulary: {list(vocab.keys())}")
    print(f"Config: d_model={config['d_model']}, decoder_layers={config['decoder_num_layers']}")

    # Load input
    images = torch.load(args.images, map_location=device)
    print(f"Input: {images.shape}")

    print("\nReady for inference. Full pipeline requires:")
    print("  1. Backbone forward (ONNX/TRT or PyTorch)")
    print("  2. Scorer forward (PyTorch)")
    print("  3. Post-processing (trajectory selection)")
    print("\nBuild TRT engine on Jetson:")
    print("  /usr/src/tensorrt/bin/trtexec \\")
    print("    --onnx=backbone_nchw.onnx \\")
    print("    --saveEngine=backbone_fp16.trt \\")
    print("    --fp16 --memPoolSize=workspace:4096MiB")


if __name__ == '__main__':
    main()
