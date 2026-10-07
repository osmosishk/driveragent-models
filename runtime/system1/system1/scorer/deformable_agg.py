"""
DeformableFeatureAggregation for the V2 scorer.

Ported from SparseDriveV2 blocks.py:22-287.
Uses CUDA ops when available, falls back to grid_sample.
Adapted for 6-camera nuScenes setup.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .keypoints_generator import SparsePoint3DKeyPointsGenerator

try:
    from system1.ops.deformable_aggregation import deformable_aggregation_ext
    if deformable_aggregation_ext is not None:
        from system1.ops import deformable_aggregation_func as DAF
    else:
        DAF = None
except ImportError:
    DAF = None

try:
    from system1.ops import feature_maps_format
except ImportError:
    feature_maps_format = None


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        config,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        num_pts: int = 8,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_deformable_func: bool = False,
        use_camera_embed: bool = True,
        residual_mode: str = "add",
        filter_outlier: bool = True,
    ):
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims must be divisible by num_groups, "
                             f"but got {embed_dims} and {num_groups}")
        self.config = config
        self.group_dims = embed_dims // num_groups
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.filter_outlier = filter_outlier

        self.proj_drop = nn.Dropout(proj_drop)

        self.kps_generator = SparsePoint3DKeyPointsGenerator(
            embed_dims=embed_dims,
            num_sample=num_pts,
            num_learnable_pts=config.num_learnable_pts,
            fix_height=tuple(config.fix_height),
            ground_height=0,
        )
        self.num_pts = self.kps_generator.num_pts

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

        self.init_weight()

    def init_weight(self):
        nn.init.constant_(self.weights_fc.weight, 0)
        nn.init.constant_(self.weights_fc.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: Optional[torch.Tensor],
        feature_maps,
        metas: dict,
        depth_prob=None,
    ):
        """
        Args:
            instance_feature: [B, num_anchor, embed_dims]
            anchor: [B, num_anchor, num_sample*2] flattened 2D waypoints
            anchor_embed: optional positional embedding (unused for scorer, pass None)
            feature_maps: list of [B, num_cams, C, H, W] or pre-formatted tuple
            metas: dict with "projection_mat" [B, 6, 4, 4], "image_wh" [B, 6, 2]
            depth_prob: unused for scorer (pass None)

        Returns:
            output: [B, num_anchor, embed_dims]
        """
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)

        if self.use_deformable_func:
            features = self._forward_cuda(
                instance_feature, anchor_embed, key_points,
                feature_maps, metas, depth_prob, bs, num_anchor,
            )
        else:
            features = self._forward_grid_sample(
                instance_feature, anchor_embed, key_points,
                feature_maps if isinstance(feature_maps, list) else metas.get("_raw_feature_maps", feature_maps),
                metas, bs, num_anchor,
            )

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _forward_cuda(self, instance_feature, anchor_embed, key_points,
                      feature_maps, metas, depth_prob, bs, num_anchor):
        """CUDA deformable aggregation path."""
        points_2d, depth, mask = self.project_points(
            key_points, metas["projection_mat"], metas.get("image_wh"),
        )
        weights = self._get_weights(instance_feature, anchor_embed, metas, mask)

        points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
            bs, num_anchor * self.num_pts, -1, 2
        )
        weights = (
            weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(
                bs, num_anchor * self.num_pts,
                self.num_cams, self.num_levels, self.num_groups,
            )
        )
        # feature_maps here should be the pre-formatted deform_value tuple
        features = DAF(*feature_maps, points_2d, weights)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        features = features.sum(dim=2)
        return features

    def _forward_grid_sample(self, instance_feature, anchor_embed, key_points,
                             feature_maps, metas, bs, num_anchor):
        """Pure PyTorch grid_sample fallback."""
        # If feature_maps is a formatted tuple, we need the raw list
        if isinstance(feature_maps, (tuple, list)) and len(feature_maps) == 3 and isinstance(feature_maps[0], torch.Tensor):
            # This is a pre-formatted tuple, we need raw feature maps
            # Try to get from metas
            raw_fmaps = metas.get("_raw_feature_maps")
            if raw_fmaps is not None:
                feature_maps = raw_fmaps
            else:
                raise ValueError("grid_sample fallback needs raw feature maps in metas['_raw_feature_maps']")

        points_2d, depth, mask = self.project_points(
            key_points, metas["projection_mat"], metas.get("image_wh"),
        )
        weights = self._get_weights(instance_feature, anchor_embed, metas, mask)

        # Sample features from each level using grid_sample
        # points_2d: [B, num_cams, num_anchor, num_pts, 2] in [0,1]
        # Convert to grid_sample format [-1, 1]
        grid = points_2d * 2 - 1  # [B, num_cams, num_anchor, num_pts, 2]

        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]

        # grid: [B*num_cams, num_anchor*num_pts, 1, 2]
        grid_flat = grid.flatten(0, 1)  # [B*num_cams, num_anchor, num_pts, 2]
        grid_flat = grid_flat.reshape(bs * num_cams, num_anchor * self.num_pts, 1, 2)

        sampled = []
        for lvl_feat in feature_maps:
            # lvl_feat: [B, num_cams, C, H, W]
            feat_flat = lvl_feat.flatten(0, 1)  # [B*num_cams, C, H, W]
            s = F.grid_sample(feat_flat, grid_flat, mode='bilinear',
                              padding_mode='zeros', align_corners=False)
            # s: [B*num_cams, C, num_anchor*num_pts, 1]
            sampled.append(s.squeeze(-1))  # [B*num_cams, C, num_anchor*num_pts]

        # Stack levels: [B*num_cams, num_levels, C, num_anchor*num_pts]
        sampled = torch.stack(sampled, dim=1)
        # Reshape: [B, num_cams, num_levels, C, num_anchor, num_pts]
        sampled = sampled.reshape(bs, num_cams, num_levels, self.embed_dims, num_anchor, self.num_pts)
        # Reorder: [B, num_anchor, num_cams, num_levels, num_pts, C]
        sampled = sampled.permute(0, 4, 1, 2, 5, 3)

        # Apply weights: [B, num_anchor, num_cams, num_levels, num_pts, num_groups]
        # Group the channels
        sampled = sampled.reshape(
            bs, num_anchor, num_cams, num_levels, self.num_pts,
            self.num_groups, self.group_dims
        )
        features = (weights[..., None] * sampled).sum(dim=2).sum(dim=2)
        # [B, num_anchor, num_pts, num_groups, group_dims]
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        features = features.sum(dim=2)
        return features

    def _get_weights(self, instance_feature, anchor_embed, metas, mask=None):
        bs, num_anchor = instance_feature.shape[:2]
        if anchor_embed is not None:
            feature = instance_feature + anchor_embed
        else:
            feature = instance_feature

        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = self.weights_fc(feature)

        if mask is not None and self.filter_outlier:
            mask = mask.permute(0, 2, 1, 3)[..., None, :, None]
            weights = weights.reshape(
                bs, num_anchor, self.num_cams,
                self.num_levels, self.num_pts, self.num_groups,
            )
            weights = weights.masked_fill(
                torch.logical_and(~mask, mask.sum(dim=2, keepdim=True) != 0),
                float("-inf"),
            )

        weights = (
            weights.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs, num_anchor, self.num_cams,
                self.num_levels, self.num_pts, self.num_groups,
            )
        )

        if self.training and self.attn_drop > 0:
            drop_mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1,
                device=weights.device, dtype=weights.dtype,
            )
            weights = ((drop_mask > self.attn_drop) * weights) / (1 - self.attn_drop)

        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """Project 3D keypoints to 2D image coordinates.

        Args:
            key_points: [B, num_anchor, num_pts, 3]
            projection_mat: [B, num_cams, 4, 4]
            image_wh: [B, num_cams, 2]

        Returns:
            points_2d: [B, num_cams, num_anchor, num_pts, 2] in [0,1]
            depth: [B, num_cams, num_anchor, num_pts]
            mask: [B, num_cams, num_anchor, num_pts] bool
        """
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # [B, num_cams, num_anchor, num_pts, 4] = [B, num_cams, 1, 1, 4, 4] @ [B, 1, num_anchor, num_pts, 4, 1]
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        mask = depth > 1e-5
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        mask = mask & (points_2d[..., 0] > 0) & (points_2d[..., 1] > 0)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
            mask = mask & (points_2d[..., 0] < 1) & (points_2d[..., 1] < 1)
        return points_2d, depth, mask
