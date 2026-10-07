"""
SparsePoint3DKeyPointsGenerator for scorer deformable aggregation.

Ported from SparseDriveV2 blocks.py:288-373.
Treats anchors as 2D waypoint sequences (x, y) and generates 3D keypoints
at multiple heights with optional learnable offsets.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class SparsePoint3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 40,
        num_learnable_pts: int = 2,
        fix_height: Tuple = (0., -0.25, -0.5, 0.25, 0.5),
        ground_height: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts

        if self.num_learnable_pts > 0:
            self.num_pts = num_sample * len(fix_height) * num_learnable_pts
            self.learnable_fc = nn.Linear(self.embed_dims, self.num_pts * 2)
        else:
            self.num_pts = num_sample * len(fix_height)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height
        self.init_weight()

    def init_weight(self):
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.constant_(self.learnable_fc.bias, 0)

    def forward(self, anchor, instance_feature=None):
        """
        Args:
            anchor: [B, num_anchor, num_sample*2] flattened 2D waypoints
            instance_feature: [B, num_anchor, embed_dims]

        Returns:
            key_points: [B, num_anchor, num_pts, 3] 3D keypoints
        """
        bs, num_anchor, _ = anchor.shape
        # Reshape to waypoints: [B, num_anchor, num_sample, 2]
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)

        if self.num_learnable_pts > 0:
            # Learnable 2D offsets: [B, num_anchor, num_sample, num_heights, num_learnable, 2]
            offset = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_sample,
                         len(self.fix_height), self.num_learnable_pts, 2)
            )
            key_points = offset + key_points[..., None, None, :]
        else:
            key_points = key_points[..., None, None, :]

        # Add z=ground_height: [B, num_anchor, num_sample, num_heights, num_learnable, 3]
        key_points = torch.cat([
            key_points,
            key_points.new_full(key_points.shape[:-1] + (1,), fill_value=self.ground_height),
        ], dim=-1)

        # Add height offsets
        fix_height = key_points.new_tensor(self.fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]

        # Flatten: [B, num_anchor, num_pts, 3]
        key_points = key_points.flatten(2, 4)
        return key_points
