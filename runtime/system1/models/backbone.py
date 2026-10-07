"""
Pure PyTorch ResNet + FPN backbone for SparseDrive.

Supports ResNet-50 (default) and ResNet-101 (--backbone resnet101).
Both output the same stage channels [256, 512, 1024, 2048] so the
FPN and detection head need no changes.

Input: (B, 6, 3, H, W) - 6 camera images
Output: List of 4 feature maps at different scales
  - Level 0: (B, 6, 256, H/8, W/8)
  - Level 1: (B, 6, 256, H/16, W/16)
  - Level 2: (B, 6, 256, H/32, W/32)
  - Level 3: (B, 6, 256, H/64, W/64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torch.utils.checkpoint import checkpoint
from typing import List
import numpy as np
from PIL import Image as PILImage


class GridMask(nn.Module):
    """GridMask augmentation matching original SparseDrive implementation.

    Creates a grid-pattern mask and drops out masked regions during training.
    Uses PIL rotation for the mask (matching original numpy/PIL pipeline).

    Args:
        use_h: Apply horizontal grid lines.
        use_w: Apply vertical grid lines.
        rotate: Max rotation angle (degrees) for the mask.
        offset: If True, fill masked regions with random values instead of 0.
        ratio: Line width as fraction of grid spacing.
        mode: 0 = keep grid lines, 1 = drop grid lines (inverted).
        prob: Probability of applying the mask per forward call.
    """

    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False,
                 ratio=0.5, mode=0, prob=1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self_l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self_l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self_l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = PILImage.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2: (hh - h) // 2 + h,
            (ww - w) // 2: (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask.copy()).float().to(x.device)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(
                2 * (np.random.rand(h, w) - 0.5)
            ).float().to(x.device)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Takes multi-scale features from ResNet and produces
    unified 256-channel features at each scale.
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Output convolutions (3x3 conv to smooth features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from ResNet [C2, C3, C4, C5]
                     Channels: [256, 512, 1024, 2048]
        
        Returns:
            List of FPN features [P2, P3, P4, P5], all with 256 channels
        """
        # Apply lateral convolutions
        laterals = [
            lateral_conv(f) 
            for f, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level and add to lower level
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # Apply output convolutions
        outputs = [
            fpn_conv(lateral)
            for lateral, fpn_conv in zip(laterals, self.fpn_convs)
        ]
        
        return outputs


class PureResNetBackbone(nn.Module):
    """
    Pure PyTorch ResNet backbone (supports ResNet-50 and ResNet-101).
    Outputs feature maps from all 4 stages.
    Both depths output [256, 512, 1024, 2048] channels.
    """

    def __init__(self, pretrained_path=None, with_cp=True, depth=50):
        super().__init__()
        if depth == 101:
            resnet = resnet101(weights=None)
        else:
            resnet = resnet50(weights=None)
        self.depth = depth

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.with_cp = with_cp

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("fc.")
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained backbone from {path}")

    def _forward_layer(self, layer, x):
        if self.with_cp and self.training:
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self._forward_layer(self.layer1, x)
        c2 = self._forward_layer(self.layer2, c1)
        c3 = self._forward_layer(self.layer3, c2)
        c4 = self._forward_layer(self.layer4, c3)

        return (c1, c2, c3, c4)


# Backward compat alias
PureResNet50Backbone = PureResNetBackbone


class SparseDriveBackbone(nn.Module):
    """
    Complete SparseDrive backbone: ResNet + FPN.

    Supports ResNet-50 (default) and ResNet-101 (depth=101).
    Both produce identical FPN output shapes.

    Can load weights from:
    1. SparseDrive checkpoint (sparsedrive_stage2.pth) - contains img_backbone and img_neck
    2. ImageNet pretrained ResNet (resnet50-19c8e357.pth) - backbone only
    """

    def __init__(self, pretrained_path: str = None, sparsedrive_ckpt: str = None, depth: int = 50):
        super().__init__()

        self.depth = depth
        # ResNet backbone (don't load pretrained yet, will load from SparseDrive ckpt)
        self.resnet = PureResNetBackbone(pretrained_path=None, with_cp=False, depth=depth)

        # FPN: channels from ResNet stages [C2, C3, C4, C5] = [256, 512, 1024, 2048]
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)

        # GridMask augmentation (applied during training only)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False,
            ratio=0.5, mode=1, prob=0.7,
        )

        # Image normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load weights
        if sparsedrive_ckpt is not None:
            self._load_sparsedrive_weights(sparsedrive_ckpt)
        elif pretrained_path is not None:
            self.resnet._load_pretrained(pretrained_path)
    
    def _load_sparsedrive_weights(self, ckpt_path: str):
        """Load backbone and FPN weights from SparseDrive checkpoint."""
        print(f"Loading backbone+FPN from SparseDrive checkpoint: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Load ResNet backbone (img_backbone.* -> resnet.*)
        backbone_sd = {}
        for k, v in state_dict.items():
            if k.startswith('img_backbone.'):
                new_key = k.replace('img_backbone.', '')
                backbone_sd[new_key] = v
        
        if backbone_sd:
            missing, unexpected = self.resnet.load_state_dict(backbone_sd, strict=False)
            print(f"  Loaded {len(backbone_sd)} backbone weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
        
        # Load FPN (img_neck.* -> fpn.*)
        # Key mapping:
        #   img_neck.lateral_convs.0.conv.weight -> fpn.lateral_convs.0.weight
        #   img_neck.fpn_convs.0.conv.weight -> fpn.fpn_convs.0.weight
        fpn_sd = {}
        for k, v in state_dict.items():
            if k.startswith('img_neck.'):
                # Remove img_neck. prefix
                new_key = k.replace('img_neck.', '')
                # Remove .conv from the middle (lateral_convs.0.conv.weight -> lateral_convs.0.weight)
                new_key = new_key.replace('.conv.', '.')
                fpn_sd[new_key] = v
        
        if fpn_sd:
            missing, unexpected = self.fpn.load_state_dict(fpn_sd, strict=False)
            print(f"  Loaded {len(fpn_sd)} FPN weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
    
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from multi-camera images.
        
        Args:
            images: (B, num_cams, 3, H, W) multi-camera images
                   Values should be in [0, 1] range OR already normalized
        
        Returns:
            List of 4 feature tensors:
              - (B, num_cams, 256, H/8, W/8)
              - (B, num_cams, 256, H/16, W/16)
              - (B, num_cams, 256, H/32, W/32)
              - (B, num_cams, 256, H/64, W/64)
        """
        B, num_cams, C, H, W = images.shape
        
        # Reshape to process all cameras together
        # (B, num_cams, 3, H, W) -> (B*num_cams, 3, H, W)
        x = images.view(B * num_cams, C, H, W)
        
        # Normalize if input is in [0, 1] range
        # (skip if already normalized - check if values are roughly in normalized range)
        if x.min() >= -0.1 and x.max() <= 1.1:
            if x.max() > 0.5:  # Likely in [0, 1] range, needs normalization
                x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        # GridMask augmentation (training only, applied after normalization)
        if self.training:
            x = self.grid_mask(x)

        # Extract ResNet features [C2, C3, C4, C5]
        resnet_features = self.resnet(x)
        
        # Apply FPN -> [P2, P3, P4, P5]
        fpn_features = self.fpn(list(resnet_features))
        
        # Reshape back to (B, num_cams, C, H, W)
        output_features = []
        for feat in fpn_features:
            _, C_feat, H_feat, W_feat = feat.shape
            feat = feat.view(B, num_cams, C_feat, H_feat, W_feat)
            output_features.append(feat)
        
        return output_features


def test_backbone(sparsedrive_ckpt=None, pretrained_path=None):
    """Test the backbone with dummy input."""
    print("Testing SparseDrive Backbone...")
    
    # Create backbone
    backbone = SparseDriveBackbone(
        pretrained_path=pretrained_path,
        sparsedrive_ckpt=sparsedrive_ckpt
    )
    backbone.eval()
    
    # Create dummy input: (B=1, 6 cameras, 3 channels, 256 height, 704 width)
    dummy_images = torch.randn(1, 6, 3, 256, 704)
    
    # Forward pass
    with torch.no_grad():
        features = backbone(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"\nOutput feature shapes:")
    for i, feat in enumerate(features):
        print(f"  Level {i}: {feat.shape}")
    
    print("\n✅ Backbone test passed!")
    
    return backbone, features


if __name__ == '__main__':
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_backbone(sparsedrive_ckpt=ckpt_path)