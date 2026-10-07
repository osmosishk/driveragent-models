"""
ConvNeXt V2 + FPN backbone for SparseDrive.

Uses timm's ConvNeXt V2 implementation with features_only mode.
Supports ConvNeXt V2 Tiny (default), Small, and Base.

Channel outputs differ from ResNet:
  - Tiny/Small: [96, 192, 384, 768]
  - Base: [128, 256, 512, 1024]

The FPN normalizes all to 256 channels, so downstream heads are unchanged.

Input: (B, 6, 3, H, W) - 6 camera images
Output: List of 4 feature maps at different scales
  - Level 0: (B, 6, 256, H/4, W/4)
  - Level 1: (B, 6, 256, H/8, W/8)
  - Level 2: (B, 6, 256, H/16, W/16)
  - Level 3: (B, 6, 256, H/32, W/32)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import List

import timm

# Import shared components from the original models/ directory
# Use importlib to avoid circular import when models_convnext is also on sys.path
import importlib.util
_MODELS_BACKBONE = str(Path(__file__).parent.parent / 'models' / 'backbone.py')
_spec = importlib.util.spec_from_file_location('models_backbone', _MODELS_BACKBONE)
_models_backbone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models_backbone)
FPN = _models_backbone.FPN
GridMask = _models_backbone.GridMask


class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXt V2 backbone using timm's features_only mode.
    Outputs feature maps from all 4 stages.

    Supports:
      - tiny:  28M params, channels [96, 192, 384, 768]
      - small: 50M params, channels [96, 192, 384, 768]
      - base:  89M params, channels [128, 256, 512, 1024]
    """

    SIZE_MAP = {
        'tiny':  ('convnextv2_tiny',  [96, 192, 384, 768]),
        'small': ('convnextv2_small', [96, 192, 384, 768]),
        'base':  ('convnextv2_base',  [128, 256, 512, 1024]),
    }

    def __init__(self, size='tiny', pretrained_path=None, with_cp=True):
        super().__init__()
        if size not in self.SIZE_MAP:
            raise ValueError(f"Unknown ConvNeXt V2 size '{size}'. Choose from: {list(self.SIZE_MAP.keys())}")

        model_name, self.stage_channels = self.SIZE_MAP[size]
        self.size = size

        # Create model with features_only=True for multi-scale feature extraction
        self.model = timm.create_model(model_name, pretrained=False, features_only=True)

        # Gradient checkpointing
        if with_cp:
            self.model.set_grad_checkpointing(enable=True)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """Load pretrained weights from a local .pt/.pth file."""
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter out classifier head keys
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('head.') and not k.startswith('norm.')
            and 'classifier' not in k
        }

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained ConvNeXt V2 ({self.size}) from {path}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images

        Returns:
            Tuple of 4 feature maps at strides [4, 8, 16, 32]
        """
        features = self.model(x)
        return tuple(features)


# Backward compat alias
ConvNeXtV2TinyBackbone = ConvNeXtV2Backbone


class SparseDriveConvNeXtBackbone(nn.Module):
    """
    Complete SparseDrive backbone: ConvNeXt V2 + FPN.

    Drop-in replacement for SparseDriveBackbone with identical interface.
    The FPN adapts to ConvNeXt's channel dimensions automatically.

    Can load weights from:
    1. ImageNet pretrained ConvNeXt V2 (for from-scratch training)
    2. SparseDrive ConvNeXt checkpoint (for resuming/evaluation)
    """

    def __init__(self, pretrained_path: str = None, sparsedrive_ckpt: str = None, size: str = 'tiny',
                 depth_supervision: bool = False):
        super().__init__()

        self.size = size

        # ConvNeXt V2 backbone
        self.convnext = ConvNeXtV2Backbone(size=size, pretrained_path=None, with_cp=True)

        # FPN: adapts to ConvNeXt channel dimensions
        self.fpn = FPN(self.convnext.stage_channels, out_channels=256)

        # GridMask augmentation (applied during training only)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False,
            ratio=0.5, mode=1, prob=0.7,
        )

        # Image normalization (ImageNet stats — same as ResNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Optional depth supervision (merged into backbone for single DDP wrapper)
        self.depth_head = None
        if depth_supervision:
            from depth_net import DenseDepthNet
            self.depth_head = DenseDepthNet(
                embed_dims=256, num_depth_layers=3,
                equal_focal=100.0, max_depth=60.0, loss_weight=0.2
            )

        # Load weights
        if sparsedrive_ckpt is not None:
            self._load_sparsedrive_weights(sparsedrive_ckpt)
        elif pretrained_path is not None:
            self.convnext._load_pretrained(pretrained_path)

    def _load_sparsedrive_weights(self, ckpt_path: str):
        """Load backbone and FPN weights from a SparseDrive ConvNeXt checkpoint."""
        print(f"Loading ConvNeXt backbone+FPN from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        # Load ConvNeXt backbone weights (convnext.* keys)
        backbone_sd = {}
        for k, v in state_dict.items():
            if k.startswith('convnext.'):
                backbone_sd[k[len('convnext.'):]] = v
        if backbone_sd:
            missing, unexpected = self.convnext.load_state_dict(backbone_sd, strict=False)
            print(f"  Loaded {len(backbone_sd)} backbone weights (missing: {len(missing)}, unexpected: {len(unexpected)})")

        # Load FPN weights (fpn.* keys)
        fpn_sd = {}
        for k, v in state_dict.items():
            if k.startswith('fpn.'):
                fpn_sd[k[len('fpn.'):]] = v
        if fpn_sd:
            missing, unexpected = self.fpn.load_state_dict(fpn_sd, strict=False)
            print(f"  Loaded {len(fpn_sd)} FPN weights (missing: {len(missing)}, unexpected: {len(unexpected)})")

    def forward(self, images: torch.Tensor, gt_depths=None, focal=None) -> List[torch.Tensor]:
        """
        Extract multi-scale features from multi-camera images.

        Args:
            images: (B, num_cams, 3, H, W) multi-camera images
                   Values should be in [0, 1] range OR already normalized
            gt_depths: optional list of 3 sparse depth maps for depth supervision
            focal: optional (B*num_cams,) focal lengths for depth supervision

        Returns:
            List of 4 feature tensors (or tuple of (features, depth_loss) if depth active)
        """
        B, num_cams, C, H, W = images.shape

        # Reshape to process all cameras together
        x = images.view(B * num_cams, C, H, W)

        # Normalize if input is in [0, 1] range
        if x.min() >= -0.1 and x.max() <= 1.1:
            if x.max() > 0.5:
                x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        # GridMask augmentation (training only)
        if self.training:
            x = self.grid_mask(x)

        # Extract ConvNeXt V2 features (4 stages)
        convnext_features = self.convnext(x)

        # Apply FPN
        fpn_features = self.fpn(list(convnext_features))

        # Depth supervision (always run depth_head if it exists to keep DDP graph constant)
        depth_loss = None
        if self.depth_head is not None and self.training:
            depth_feats = [f for f in fpn_features[:3]]
            if gt_depths is not None and focal is not None:
                depth_loss = self.depth_head(depth_feats, focal, gt_depths)
            else:
                # No GT depths this batch — still run forward for DDP consistency, discard
                dummy_loss = sum(p.sum() * 0.0 for p in self.depth_head.parameters())
                depth_loss = dummy_loss

        # Reshape back to (B, num_cams, C, H, W)
        output_features = []
        for feat in fpn_features:
            _, C_feat, H_feat, W_feat = feat.shape
            feat = feat.view(B, num_cams, C_feat, H_feat, W_feat)
            output_features.append(feat)

        if depth_loss is not None:
            return output_features, depth_loss
        return output_features


def test_backbone(pretrained_path=None, size='tiny'):
    """Test the ConvNeXt V2 backbone with dummy input."""
    print(f"Testing SparseDrive ConvNeXt V2 ({size}) Backbone...")

    backbone = SparseDriveConvNeXtBackbone(
        pretrained_path=pretrained_path,
        size=size,
    )
    backbone.eval()

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    convnext_params = sum(p.numel() for p in backbone.convnext.parameters())
    fpn_params = sum(p.numel() for p in backbone.fpn.parameters())
    print(f"\nParameters: {total_params/1e6:.1f}M total ({convnext_params/1e6:.1f}M backbone, {fpn_params/1e6:.1f}M FPN)")

    # Create dummy input: (B=1, 6 cameras, 3 channels, 256 height, 704 width)
    dummy_images = torch.randn(1, 6, 3, 256, 704)

    with torch.no_grad():
        features = backbone(dummy_images)

    print(f"\nInput shape: {dummy_images.shape}")
    print(f"\nOutput feature shapes:")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape}")

    print("\nBackbone test passed!")
    return backbone, features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default='tiny', choices=['tiny', 'small', 'base'])
    parser.add_argument('--pretrained', type=str, default=None)
    args = parser.parse_args()
    test_backbone(pretrained_path=args.pretrained, size=args.size)
