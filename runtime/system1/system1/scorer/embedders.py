import torch.nn as nn


class PathEmbedder(nn.Module):
    """Embed path anchor coordinates into d_model space."""

    def __init__(self, len_path, d_ffn, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len_path * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, path_anchors):
        # path_anchors: [B, num_paths, len_path, 3]
        return self.mlp(path_anchors.flatten(-2, -1))  # [B, num_paths, d_model]


class VelocityEmbedder(nn.Module):
    """Embed velocity profile into d_model space."""

    def __init__(self, len_vel, d_ffn, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len_vel, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, vel_anchors):
        # vel_anchors: [B, num_vels, len_vel]
        return self.mlp(vel_anchors)  # [B, num_vels, d_model]
