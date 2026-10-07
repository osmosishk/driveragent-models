"""
System 1 Model: Shared backbone + V2 factorized scorer + optional perception heads.

Outputs:
1. Selected trajectory + 8 confidence scores (for driving)
2. Agent/map features (for System 2 consumption, optional)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import System1Config
from .scorer import FactorizedScorer

# Import ConvNeXt V2 backbone from existing codebase
_MODELS_CONVNEXT_DIR = str(Path(__file__).parent.parent / 'models_convnext')
if _MODELS_CONVNEXT_DIR not in sys.path:
    sys.path.insert(0, _MODELS_CONVNEXT_DIR)
from backbone import SparseDriveConvNeXtBackbone


class System1Model(nn.Module):
    """
    System 1: Shared backbone + perception + V2 factorized scorer.
    """

    def __init__(self, config: System1Config, backbone_ckpt: str = None):
        super().__init__()
        self.config = config

        # Shared backbone (existing ConvNeXt V2 Tiny + FPN)
        self.backbone = SparseDriveConvNeXtBackbone(
            size='tiny',
            depth_supervision=False,
        )

        # Load backbone weights if provided
        if backbone_ckpt is not None:
            self._load_backbone(backbone_ckpt)

        # V2 factorized scorer (NEW)
        self.scorer = FactorizedScorer(config)

        # Perception heads (for System 2, optional initially)
        self.detection_head = None
        self.map_head = None

    def _load_backbone(self, ckpt_path):
        """Load backbone weights from a checkpoint."""
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Handle different checkpoint formats
        if 'backbone_state_dict' in state:
            backbone_state = state['backbone_state_dict']
        elif 'backbone' in state:
            backbone_state = state['backbone']
        elif 'model' in state:
            backbone_state = {k.replace('backbone.', ''): v
                              for k, v in state['model'].items()
                              if k.startswith('backbone.')}
        else:
            backbone_state = state

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            print(f"Backbone: {len(missing)} missing keys (expected for non-backbone params)")
        if unexpected:
            print(f"Backbone: {len(unexpected)} unexpected keys")

    def forward(self, images, ego_state, camera_metas, targets=None,
                scene_ctx=None):
        """
        Args:
            images: [B, 6, 3, H, W] multi-view camera images (normalized)
            ego_state: [B, ego_state_dim] ego vehicle state
            camera_metas: dict with "projection_mat" [B, 6, 4, 4], "image_wh" [B, 6, 2]
            targets: dict with GT trajectory, path, velocity (training only)
            scene_ctx: optional dict with GT or detected agent/map context

        Returns:
            output: dict with:
                "trajectory": [B, T, 3] selected trajectory
                "agent_features": [B, N, 256] (for System 2, None if no det head)
                "map_features": [B, M, 256] (for System 2, None if no map head)
            loss_dict: dict of losses (training only)
        """
        # Backbone forward (shared)
        fpn_features = self.backbone(images)

        # Scorer forward (V2 factorized)
        scorer_output, loss_dict = self.scorer(
            fpn_features, ego_state, camera_metas, targets,
            scene_ctx=scene_ctx,
        )

        # Perception forward (for System 2, optional)
        agent_features = None
        map_features = None
        if self.detection_head is not None:
            agent_features = self.detection_head(fpn_features, camera_metas)
        if self.map_head is not None:
            map_features = self.map_head(fpn_features, camera_metas)

        output = {
            **scorer_output,
            "agent_features": agent_features,
            "map_features": map_features,
        }

        return output, loss_dict

    def freeze_backbone(self):
        """Freeze backbone for scorer-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Unfreeze backbone for joint training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
