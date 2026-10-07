"""Deterministic action sampling for DTCP — uses the Beta mode instead of
the PyTorch Beta.rsample() call in TCP.get_action.

Why mode (not sample): DTCP was trained with Beta(α=m·s+1, β=(1-m)·s+1)
where m is the ground-truth action mapped to [0,1]. By construction the
mode of Beta exactly equals the human action, so the mode is the right
deterministic deployment policy.

Beta mode formula (for α, β > 1): (α - 1) / (α + β - 2).
"""
from __future__ import annotations

import numpy as np


def beta_mode_action(mu: np.ndarray, sigma: np.ndarray) -> tuple:
    """Replicate the deterministic part of TCP.get_action with the mode.

    Args:
        mu:    (..., 2) — Beta alpha for (acc, steer). Required: > 1.
        sigma: (..., 2) — Beta beta  for (acc, steer). Required: > 1.

    Returns:
        (throttle, steer, brake) — each scalar float in PyTorch's convention:
            throttle in [0, 1], brake in [0, 1] (mutually exclusive),
            steer    in [-1, 1].
    """
    mu = np.asarray(mu).reshape(-1)
    sigma = np.asarray(sigma).reshape(-1)
    if mu.shape != (2,) or sigma.shape != (2,):
        raise ValueError(f"expected mu/sigma shape (2,), got {mu.shape}/{sigma.shape}")

    # Mode of Beta(α, β) for α, β > 1.
    mode = (mu - 1.0) / (mu + sigma - 2.0)
    # Map [0, 1] -> [-1, 1] to match TCP's action space.
    action = mode * 2.0 - 1.0
    action = np.clip(action, -1.0, 1.0)
    acc, steer = float(action[0]), float(action[1])

    throttle = max(acc, 0.0)
    brake = max(-acc, 0.0)
    return throttle, steer, brake
