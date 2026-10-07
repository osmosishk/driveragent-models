import numpy as np
import torch
import torch.nn as nn


class TrajectoryVocabulary(nn.Module):
    """Frozen trajectory vocabulary loaded from K-Means output."""

    def __init__(self, config):
        super().__init__()
        path_data = np.load(config.path_anchor_file)           # [1024, 40, 3]
        vel_data = np.load(config.velocity_anchor_file)         # [256, 6]
        traj_data = np.load(config.trajectory_anchor_file)      # npz

        self.register_buffer("path_anchors",
            torch.from_numpy(path_data).float())                # [1024, 40, 3]
        self.register_buffer("vel_anchors",
            torch.from_numpy(vel_data).float())                 # [256, 6]
        self.register_buffer("traj_anchors",
            torch.from_numpy(traj_data["trajectory"]).float())  # [1024, 256, 6, 3]
        self.register_buffer("traj_mask",
            torch.from_numpy(traj_data["trajectory_mask"]).float())  # [1024, 256, 6]

    def get_batch(self, batch_size):
        """Expand vocabulary for batch processing."""
        return {
            "path": self.path_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1),
            "vel": self.vel_anchors.unsqueeze(0).expand(batch_size, -1, -1),
            "traj": self.traj_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
            "traj_mask": self.traj_mask.unsqueeze(0).expand(batch_size, -1, -1, -1),
        }
