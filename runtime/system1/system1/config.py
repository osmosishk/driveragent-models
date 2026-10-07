from dataclasses import dataclass, field
from typing import List


@dataclass
class System1Config:
    """System 1 configuration for nuScenes factorized trajectory scoring."""

    # === Backbone (existing ConvNeXt V2 Tiny + FPN) ===
    backbone: str = "convnext_v2_tiny"
    fpn_channels: int = 256
    num_levels: int = 4

    # === Cameras (nuScenes) ===
    num_cams: int = 6
    img_size: List[int] = field(default_factory=lambda: [256, 704])

    # === Scorer ===
    d_model: int = 256
    d_ffn: int = 1024
    num_head: int = 8
    dropout: float = 0.0

    # === Vocabulary ===
    path_anchor_file: str = "ckpt/kmeans/nuscenes_path_1024.npy"
    velocity_anchor_file: str = "ckpt/kmeans/nuscenes_velocity_256.npy"
    trajectory_anchor_file: str = "ckpt/kmeans/nuscenes_trajectory_1024_256.npz"

    mode_path: int = 1024
    mode_vel: int = 256
    len_path: int = 30              # 15m at 0.5m interval
    len_vel_seq: int = 6            # 3s at 0.5s interval
    path_interval: float = 0.5
    vel_time_interval: float = 0.5

    # === Decoder ===
    decoder_num_layers: int = 2
    path_filter_num: List[int] = field(default_factory=lambda: [128, 20])
    velocity_filter_num: List[int] = field(default_factory=lambda: [64, 20])

    # === Loss scaling ===
    path_sigmas: float = 8.0
    velocity_sigmas: float = 6.0
    trajectory_sigmas: float = 8.0

    # === Deformable aggregation ===
    fix_height: List[float] = field(default_factory=lambda: [0., -0.25, -0.5, 0.25, 0.5])
    num_learnable_pts: int = 2

    # === Metric heads ===
    metrics: List[str] = field(default_factory=lambda: [
        "collision", "drivable_area", "direction", "traffic_light",
        "time_to_collision", "progress", "lane_keeping", "comfort"
    ])
    metric_loss_weight: float = 5.0

    # === Ego status ===
    ego_state_dim: int = 8          # vx, vy, ax, ay, yaw_rate, speed, cmd_onehot(2)
