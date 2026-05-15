"""Behavior policy baselines."""

from __future__ import annotations

import torch
from torch import Tensor


class BehaviorPolicy:
    """Small deterministic policy used for smoke data and ablations.

    The policy maps the final observation features into the action space when
    enough features are available and pads with zeros otherwise. This gives
    reproducible non-random actions without pretending to be a trained policy.
    """

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim

    def select_action(self, observation: Tensor) -> Tensor:
        flat = observation.flatten()
        action = torch.zeros(self.action_dim, dtype=observation.dtype, device=observation.device)
        n = min(self.action_dim, flat.numel())
        if n:
            action[:n] = torch.tanh(flat[-n:])
        return action
