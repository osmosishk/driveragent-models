"""
FactorizedScorer: V2-style factorized trajectory scorer.

Wires vocabulary + embedders + 2-layer coarse-to-fine decoder.
Replaces the 6-mode trajectory head entirely.
"""

import torch
import torch.nn as nn

from .vocabulary import TrajectoryVocabulary
from .embedders import PathEmbedder, VelocityEmbedder
from .decoder_layer import ScorerDecoderLayer


class FactorizedScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Frozen vocabulary
        self.vocabulary = TrajectoryVocabulary(config)

        # Embedders
        self.path_embedder = PathEmbedder(config.len_path, config.d_ffn, config.d_model)
        self.vel_embedder = VelocityEmbedder(config.len_vel_seq, config.d_ffn, config.d_model)

        # Ego status encoder
        self.status_encoder = nn.Linear(config.ego_state_dim, config.d_model)

        # Decoder layers (coarse-to-fine)
        self.decoder = nn.ModuleList()
        for i in range(config.decoder_num_layers):
            self.decoder.append(ScorerDecoderLayer(config, decoder_idx=i))

    def forward(self, fpn_features, ego_state, camera_metas, targets=None,
                scene_ctx=None):
        """
        Args:
            fpn_features: list of [B, num_cams, C, H, W] at each FPN level
            ego_state: [B, ego_state_dim]
            camera_metas: dict with "projection_mat" [B, 6, 4, 4], "image_wh" [B, 6, 2]
            targets: dict with "path", "path_mask", "velocity", "trajectory" (training only)
            scene_ctx: optional dict with "agent_ctx" [B, N, 9], "agent_mask" [B, N],
                       "map_ctx" [B, M, 40], "map_mask" [B, M]

        Returns:
            output: dict with "trajectory" [B, T, 3]
            loss_dict: dict of losses (training only)
        """
        B = ego_state.shape[0]

        # Get vocabulary for batch
        vocab = self.vocabulary.get_batch(B)

        # Embed
        path_embed = self.path_embedder(vocab["path"])      # [B, 1024, d_model]
        vel_embed = self.vel_embedder(vocab["vel"])          # [B, 256, d_model]
        status_embed = self.status_encoder(ego_state)        # [B, d_model]

        # Pass raw scene context — each decoder layer encodes with its own encoders
        encoded_scene_ctx = scene_ctx

        # Build camera_feature dict for decoder
        camera_feature = {
            "feature_maps": fpn_features,
            "projection_mat": camera_metas["projection_mat"],
            "image_wh": camera_metas.get("image_wh"),
        }

        # Run decoder layers
        feature = (path_embed, vel_embed, vocab["path"], vocab["vel"],
                   vocab["traj"], vocab["traj_mask"])

        output = {}
        loss_dict = {}
        for layer in self.decoder:
            feature, layer_output, layer_loss = layer(
                *feature, camera_feature, status_embed, targets,
                scene_ctx=encoded_scene_ctx,
            )
            output.update(layer_output)
            loss_dict.update(layer_loss)

        return output, loss_dict
