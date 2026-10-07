"""
Coarse-to-fine scorer decoder layer.

Ported from SparseDriveV2 custom_decoder.py:43-319.
Adapted for nuScenes: 6 cameras, 40-point paths, 6-step velocities.
Removed NAVSIM-specific metric caching (pdm_token_path).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_agg import DeformableFeatureAggregation
from .losses import path_loss, velocity_loss, trajectory_loss

try:
    from system1.ops.deformable_aggregation import deformable_aggregation_ext
    if deformable_aggregation_ext is not None:
        from system1.ops import deformable_format
    else:
        deformable_format = None
except ImportError:
    deformable_format = None


class ScorerDecoderLayer(nn.Module):
    def __init__(self, config, decoder_idx):
        super().__init__()
        self._config = config
        self.decoder_idx = decoder_idx
        d_model = config.d_model
        d_ffn = config.d_ffn

        # --- Path scoring ---
        self.p_deform_model = DeformableFeatureAggregation(
            config=config,
            embed_dims=d_model,
            num_groups=8,
            num_levels=config.num_levels,
            num_cams=config.num_cams,
            num_pts=config.len_path,
            attn_drop=0.0,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="add",
        )
        self.p_attention = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.p_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
        )
        self.p_norm1 = nn.LayerNorm(d_model)
        self.p_dropout1 = nn.Dropout(0.1)
        self.p_norm2 = nn.LayerNorm(d_model)
        self.p_dropout2 = nn.Dropout(0.1)
        self.path_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1),
        )

        # --- v6 pooled scene bias (loads from checkpoint) ---
        self.scene_proj = nn.Linear(d_model, d_model)
        self.scene_gate = nn.Parameter(torch.tensor(-5.0))

        # --- Per-path perception cross-attention ---
        # Each of 1024 paths attends independently to agents/map elements
        # so "path near a car" gets different context than "path in empty lane"
        self.agent_encoder = nn.Sequential(
            nn.Linear(9, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(40, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        self.path_agent_attn = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.path_norm_agent = nn.LayerNorm(d_model)
        self.path_map_attn = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.path_norm_map = nn.LayerNorm(d_model)
        # Raw zero-init gate: starts as exact no-op, learns to open gradually
        self.path_agent_gate = nn.Parameter(torch.zeros(1))
        # Velocity: global scene attention (no spatial geometry for velocity)
        self.vel_scene_attn = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.vel_norm_scene = nn.LayerNorm(d_model)
        # Zero-init out_proj so attn_out=0 at init (prevents gate gradient explosion)
        for attn in [self.path_agent_attn, self.path_map_attn, self.vel_scene_attn]:
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

        # --- Velocity scoring ---
        self.v_img_attention = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.v_attention = nn.MultiheadAttention(
            d_model, config.num_head, dropout=config.dropout, batch_first=True,
        )
        self.v_ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
        )
        self.v_norm1 = nn.LayerNorm(d_model)
        self.v_dropout1 = nn.Dropout(0.1)
        self.v_norm2 = nn.LayerNorm(d_model)
        self.v_dropout2 = nn.Dropout(0.1)
        self.vel_mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1),
        )

        # --- Trajectory reconditioning (final layer only) ---
        if self.decoder_idx == config.decoder_num_layers - 1:
            self.t_deform_model = DeformableFeatureAggregation(
                config=config,
                embed_dims=d_model,
                num_groups=8,
                num_levels=config.num_levels,
                num_cams=config.num_cams,
                num_pts=config.len_vel_seq,  # num_poses = 6 waypoints
                attn_drop=0.0,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode="add",
            )
            self.t_attention = nn.MultiheadAttention(
                d_model, config.num_head, dropout=config.dropout, batch_first=True,
            )
            self.t_ffn = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model),
            )
            self.t_norm1 = nn.LayerNorm(d_model)
            self.t_dropout1 = nn.Dropout(0.1)
            self.t_norm2 = nn.LayerNorm(d_model)
            self.t_dropout2 = nn.Dropout(0.1)
            self.traj_mlp = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1),
            )
            # 8 metric heads
            self.metric_heads = nn.ModuleDict()
            for metric in config.metrics:
                self.metric_heads[metric] = nn.Sequential(
                    nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, 1),
                )
            # Heading regression head: predicts heading per timestep from
            # selected trajectory's embedding (decoupled from xy scoring)
            self.heading_head = nn.Sequential(
                nn.Linear(d_model, d_ffn), nn.ReLU(),
                nn.Linear(d_ffn, config.len_vel_seq),  # 6 heading values
            )

    def forward(self, path_embed, vel_embed, path_vocab, vel_vocab,
                traj_vocab, traj_mask, camera_feature, status_encoding, targets,
                scene_ctx=None):
        """
        Args:
            path_embed: [B, num_path, d_model]
            vel_embed: [B, num_vel, d_model]
            path_vocab: [B, num_path, len_path, 3]
            vel_vocab: [B, num_vel, len_vel_seq]
            traj_vocab: [B, num_path, num_vel, len_vel_seq, 3]
            traj_mask: [B, num_path, num_vel, len_vel_seq]
            camera_feature: dict with "feature_maps", "projection_mat", "image_wh"
            status_encoding: [B, d_model]
            targets: dict with GT path/velocity/trajectory (training only)
            scene_ctx: optional dict with "agent_feat" [B, N, d_model], "map_feat" [B, M, d_model],
                       "scene_mask" [B, N+M] (True=valid)

        Returns:
            feature_tuple: filtered (path_embed, vel_embed, path_vocab, vel_vocab, traj_vocab, traj_mask)
            output: dict with "trajectory" if final layer
            loss_dict: dict of losses if training
        """
        num_path = path_embed.shape[1]
        num_vel = vel_embed.shape[1]

        # Prepare image features for velocity cross-attention
        # Use last FPN level flattened: [B, num_cams*H*W, d_model]
        img_value = camera_feature["feature_maps"][-1].permute(0, 1, 3, 4, 2).flatten(1, 3)

        # Prepare deformable features (CUDA format or raw for fallback)
        if deformable_format is not None:
            deform_value = deformable_format(camera_feature["feature_maps"])
        else:
            deform_value = camera_feature["feature_maps"]

        # Build metas dict for deformable agg
        metas = {
            "projection_mat": camera_feature["projection_mat"],
            "image_wh": camera_feature.get("image_wh"),
            "_raw_feature_maps": camera_feature["feature_maps"],
        }

        # Encode perception context
        if scene_ctx is not None:
            # NaN-safe: GT boxes may have NaN velocities (nuScenes characteristic)
            agent_feat = self.agent_encoder(scene_ctx["agent_ctx"].nan_to_num(0.0))  # [B, N, d_model]
            map_feat = self.map_encoder(scene_ctx["map_ctx"].nan_to_num(0.0))        # [B, M, d_model]
            agent_pad_mask = ~scene_ctx["agent_mask"]  # True=IGNORE for MHA
            map_pad_mask = ~scene_ctx["map_mask"]
            # Ensure at least one key valid per sample (prevents MHA softmax NaN)
            agent_pad_mask[:, 0] = False
            map_pad_mask[:, 0] = False
            # Combined for velocity (global scene attention)
            all_feat = torch.cat([agent_feat, map_feat], dim=1)
            all_pad_mask = torch.cat([agent_pad_mask, map_pad_mask], dim=1)
        else:
            agent_feat = map_feat = all_feat = None
            agent_pad_mask = map_pad_mask = all_pad_mask = None

        # Add ego status encoding
        path_embed = path_embed + status_encoding.unsqueeze(1)
        vel_embed = vel_embed + status_encoding.unsqueeze(1)

        # v6 pooled scene bias (loaded from checkpoint, provides working baseline)
        if agent_feat is not None:
            # Masked mean: only count valid (non-padded) elements
            valid_mask = torch.cat([scene_ctx["agent_mask"], scene_ctx["map_mask"]], dim=1)  # [B, N+M]
            valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [B, 1, 1]
            masked_feat = all_feat * valid_mask.unsqueeze(-1)  # zero out padded
            scene_feat = masked_feat.sum(dim=1) / valid_count.squeeze(1)  # [B, d_model]
            scene_bias = self.scene_gate.sigmoid() * self.scene_proj(scene_feat)  # [B, d_model]
            path_embed = path_embed + scene_bias.unsqueeze(1)
            vel_embed = vel_embed + scene_bias.unsqueeze(1)

        # --- Path scoring ---
        path_vocab_flat = path_vocab[..., :2].flatten(-2)  # [B, num_path, len_path*2]
        path_embed = self.p_deform_model(
            path_embed, path_vocab_flat, None, deform_value, metas, None,
        )
        # Per-path cross-attention to agents and map (zero-gated residual)
        # LN inside gated residual so gate=0 is exact no-op (no LN on path_embed)
        if agent_feat is not None:
            path_embed = path_embed + self.path_agent_gate * self.path_norm_agent(
                self.path_agent_attn(path_embed, agent_feat, agent_feat, key_padding_mask=agent_pad_mask)[0]
            )
            path_embed = path_embed + self.path_agent_gate * self.path_norm_map(
                self.path_map_attn(path_embed, map_feat, map_feat, key_padding_mask=map_pad_mask)[0]
            )

        path_embed = path_embed + self.p_dropout1(
            self.p_attention(path_embed, path_embed, path_embed)[0]
        )
        path_embed = self.p_norm1(path_embed)
        path_embed = path_embed + self.p_dropout2(self.p_ffn(path_embed))
        path_embed = self.p_norm2(path_embed)
        path_scores = self.path_mlp(path_embed).squeeze(-1)  # [B, num_path]

        # --- Velocity scoring ---
        # Global scene attention for velocity (no spatial structure)
        if all_feat is not None:
            vel_embed = vel_embed + self.path_agent_gate * self.vel_norm_scene(
                self.vel_scene_attn(vel_embed, all_feat, all_feat, key_padding_mask=all_pad_mask)[0]
            )
        vel_embed = vel_embed + self.v_img_attention(vel_embed, img_value, img_value)[0]
        vel_embed = vel_embed + self.v_dropout1(
            self.v_attention(vel_embed, vel_embed, vel_embed)[0]
        )
        vel_embed = self.v_norm1(vel_embed)
        vel_embed = vel_embed + self.v_dropout2(self.v_ffn(vel_embed))
        vel_embed = self.v_norm2(vel_embed)
        vel_scores = self.vel_mlp(vel_embed).squeeze(-1)  # [B, num_vel]

        # --- Coarse filter ---
        filter_traj_vocab = traj_vocab.clone()
        filter_traj_mask = traj_mask.clone()

        if num_path > self._config.path_filter_num[self.decoder_idx]:
            topk_path_scores, topk_path_indices = torch.topk(
                path_scores, self._config.path_filter_num[self.decoder_idx], dim=1
            )
            filter_path_embed = torch.gather(
                path_embed, 1,
                topk_path_indices.unsqueeze(-1).expand(-1, -1, path_embed.shape[-1])
            )
            filter_path_vocab = torch.gather(
                path_vocab, 1,
                topk_path_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, path_vocab.shape[-2], path_vocab.shape[-1]
                )
            )
            filter_traj_vocab = torch.gather(
                filter_traj_vocab, 1,
                topk_path_indices[:, :, None, None, None].expand(
                    -1, -1, filter_traj_vocab.shape[-3],
                    filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]
                )
            )
            filter_traj_mask = torch.gather(
                filter_traj_mask, 1,
                topk_path_indices[:, :, None, None].expand(
                    -1, -1, filter_traj_mask.shape[-2], filter_traj_mask.shape[-1]
                )
            )
        else:
            filter_path_embed = path_embed
            filter_path_vocab = path_vocab

        if num_vel > self._config.velocity_filter_num[self.decoder_idx]:
            topk_vel_scores, topk_vel_indices = torch.topk(
                vel_scores, self._config.velocity_filter_num[self.decoder_idx], dim=1
            )
            filter_vel_embed = torch.gather(
                vel_embed, 1,
                topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_embed.shape[-1])
            )
            filter_vel_vocab = torch.gather(
                vel_vocab, 1,
                topk_vel_indices.unsqueeze(-1).expand(-1, -1, vel_vocab.shape[-1])
            )
            filter_traj_vocab = torch.gather(
                filter_traj_vocab, 2,
                topk_vel_indices[:, None, :, None, None].expand(
                    -1, filter_traj_vocab.shape[-4], -1,
                    filter_traj_vocab.shape[-2], filter_traj_vocab.shape[-1]
                )
            )
            filter_traj_mask = torch.gather(
                filter_traj_mask, 2,
                topk_vel_indices[:, None, :, None].expand(
                    -1, filter_traj_mask.shape[-3], -1, filter_traj_mask.shape[-1]
                )
            )
        else:
            filter_vel_embed = vel_embed
            filter_vel_vocab = vel_vocab

        # --- Trajectory reconditioning (final layer only) ---
        output = {}
        if self.decoder_idx == self._config.decoder_num_layers - 1:
            # Compose path + velocity embeddings
            traj_embed = filter_path_embed.unsqueeze(2) + filter_vel_embed.unsqueeze(1)
            traj_embed = traj_embed.flatten(1, 2)  # [B, num_path*num_vel, d_model]

            # Deformable agg on composed trajectories
            filter_traj_vocab_flat = filter_traj_vocab[..., :2].flatten(1, 2).flatten(-2)
            traj_embed = self.t_deform_model(
                traj_embed, filter_traj_vocab_flat, None, deform_value, metas, None,
            )
            traj_embed = traj_embed + self.t_dropout1(
                self.t_attention(traj_embed, traj_embed, traj_embed)[0]
            )
            traj_embed = self.t_norm1(traj_embed)
            traj_embed = traj_embed + self.t_dropout2(self.t_ffn(traj_embed))
            traj_embed = self.t_norm2(traj_embed)
            traj_scores = self.traj_mlp(traj_embed).squeeze(-1)  # [B, num_trajs]

            # Metric heads
            metric_logit = {}
            for metric in self._config.metrics:
                metric_logit[metric] = self.metric_heads[metric](traj_embed).squeeze(-1)

            # Select best trajectory using metric scores
            scores = (
                metric_logit["collision"].sigmoid() *
                metric_logit["drivable_area"].sigmoid() *
                metric_logit["direction"].sigmoid() *
                metric_logit["traffic_light"].sigmoid()
            ) * (
                5 * metric_logit["time_to_collision"].sigmoid() +
                5 * metric_logit["progress"].sigmoid() +
                2 * metric_logit["lane_keeping"].sigmoid() +
                2 * metric_logit["comfort"].sigmoid()
            )
            bs_indices = torch.arange(scores.shape[0], device=scores.device)
            mode_indices = scores.argmax(1)
            selected_traj = filter_traj_vocab.flatten(1, 2)[bs_indices, mode_indices]

            # Heading regression: predict heading from selected trajectory embedding
            selected_embed = traj_embed[bs_indices, mode_indices]  # [B, d_model]
            heading_pred = self.heading_head(selected_embed)  # [B, 6]
            # Replace vocab heading with regressed heading
            selected_traj = selected_traj.clone()
            selected_traj[:, :, 2] = heading_pred
            output["trajectory"] = selected_traj  # [B, len_vel_seq, 3]

        # --- Losses ---
        loss_dict = {}
        if self.training and targets is not None:
            # Path loss
            loss_dict[f'path_loss_{self.decoder_idx}'] = path_loss(
                path_scores, path_vocab, targets["path"],
                targets["path_mask"], self._config.path_sigmas, self._config.len_path,
            )
            # Velocity loss
            loss_dict[f'velocity_loss_{self.decoder_idx}'] = velocity_loss(
                vel_scores, vel_vocab, targets["velocity"],
                self._config.velocity_sigmas,
            )
            # Trajectory loss (final layer only)
            if self.decoder_idx == self._config.decoder_num_layers - 1:
                loss_dict[f'traj_loss_{self.decoder_idx}'] = trajectory_loss(
                    traj_scores, filter_traj_vocab.flatten(1, 2),
                    targets["trajectory"], self._config.trajectory_sigmas,
                )
                # Heading regression loss (smooth L1 on per-timestep heading)
                gt_heading = targets["trajectory"][:, :, 2]  # [B, T]
                # Wrap predicted heading to [-pi, pi] relative to GT
                heading_diff = heading_pred - gt_heading
                heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
                loss_dict[f'heading_loss_{self.decoder_idx}'] = 0.5 * heading_diff.abs().mean()
                # Metric losses (if GT metrics provided)
                if "metrics" in targets:
                    for metric in self._config.metrics:
                        if metric in targets["metrics"]:
                            metric_pred = metric_logit[metric]
                            metric_gt = targets["metrics"][metric].to(metric_pred)
                            metric_gt[metric_gt == 0.5] = 0.0
                            m_loss = F.binary_cross_entropy_with_logits(metric_pred, metric_gt)
                            loss_dict[f'{metric}_loss_{self.decoder_idx}'] = m_loss * self._config.metric_loss_weight

        return (
            (filter_path_embed, filter_vel_embed, filter_path_vocab,
             filter_vel_vocab, filter_traj_vocab, filter_traj_mask),
            output,
            loss_dict,
        )
