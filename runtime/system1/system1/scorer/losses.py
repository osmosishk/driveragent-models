import math

import torch
import torch.nn.functional as F


def _stable_soft_ce(scores, neg_dist):
    """Numerically stable soft cross-entropy.

    Computes CE against softmax targets with label smoothing to prevent log(0).
    """
    # Clamp distances to prevent overflow in softmax
    neg_dist = neg_dist.clamp(min=-50.0)
    target = neg_dist.softmax(dim=1)
    # Label smoothing: mix with uniform to prevent exact zeros
    eps = 1e-4
    n = target.shape[1]
    target = (1 - eps) * target + eps / n
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(scores, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def path_loss(path_scores, path_vocab, gt_path, gt_path_mask,
              sigma=4.0, len_path=30, heading_weight=0.5):
    """
    Soft cross-entropy loss for path scoring, including heading.
    """
    diff_xy = (path_vocab - gt_path[:, None])[..., :2]        # [B, num_paths, S, 2]
    diff_h = path_vocab[..., 2:3] - gt_path[:, None, :, 2:3]  # [B, num_paths, S, 1]
    diff_h = (diff_h + math.pi) % (2 * math.pi) - math.pi     # wrap to [-pi, pi]
    diff = torch.cat([diff_xy, heading_weight * diff_h], dim=-1)  # [B, num_paths, S, 3]

    dist = diff.pow(2).sum(-1)                        # [B, num_paths, S]
    mask = gt_path_mask[:, None].float()
    dist = dist * mask
    valid_cnt = mask.sum(-1).clamp(min=1.0)
    dist = dist.sum(-1) / valid_cnt                   # [B, num_paths]
    neg_dist = -dist * sigma * len_path
    return _stable_soft_ce(path_scores, neg_dist)


def velocity_loss(vel_scores, vel_vocab, gt_velocity, sigma=4.0):
    """
    Soft cross-entropy loss for velocity scoring.
    """
    dist = (vel_vocab - gt_velocity[:, None]).abs().sum(-1)  # [B, num_vels]
    neg_dist = -dist * sigma
    return _stable_soft_ce(vel_scores, neg_dist)


def trajectory_loss(traj_scores, traj_vocab, gt_trajectory,
                    sigma=4.0, heading_weight=0.5):
    """
    Soft cross-entropy loss for fine-grained trajectory scoring, including heading.
    """
    diff_xy = (traj_vocab - gt_trajectory[:, None])[..., :2]         # [B, num_trajs, T, 2]
    diff_h = traj_vocab[..., 2:3] - gt_trajectory[:, None, :, 2:3]   # [B, num_trajs, T, 1]
    diff_h = (diff_h + math.pi) % (2 * math.pi) - math.pi            # wrap to [-pi, pi]
    diff = torch.cat([diff_xy, heading_weight * diff_h], dim=-1)      # [B, num_trajs, T, 3]

    dist = diff.pow(2).sum((-2, -1))  # [B, num_trajs]
    neg_dist = -dist * sigma
    return _stable_soft_ce(traj_scores, neg_dist)
