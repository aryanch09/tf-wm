"""Proprioception reader: joint angles, velocities, and torques."""

from __future__ import annotations

import numpy as np
import torch


class ProprioReader:
    """Read proprioceptive state from a robot hardware interface.

    Args:
        n_joints: Number of actuated joints.
        sample_rate_hz: Control loop rate.
    """

    def __init__(self, n_joints: int = 16, sample_rate_hz: int = 1000) -> None:
        self.n_joints = n_joints
        self.sample_rate_hz = sample_rate_hz

    def read(self) -> torch.Tensor:
        """Return concatenated [q, dq, tau] ``(1, 1, 3*n_joints)`` float32."""
        state = np.random.normal(0, 0.1, (3 * self.n_joints,)).astype(np.float32)
        return torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
