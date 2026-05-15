"""Model predictive control planner."""

from __future__ import annotations

import torch
from torch import Tensor


class MPCPlanner:
    """Deterministic zero-order MPC baseline.

    This planner provides a conservative baseline for tests and ablations. The
    learned CEM planner in :mod:`tfwm.planning.cem_mpc` is the stronger research
    planner; this class keeps a minimal API for environments that only need a
    stable open-loop action sequence.
    """

    def __init__(self, horizon: int = 8, action_dim: int = 4) -> None:
        self.horizon = horizon
        self.action_dim = action_dim

    def plan(self, state: Tensor, model: torch.nn.Module) -> Tensor:
        del model
        return torch.zeros((self.horizon, self.action_dim), dtype=state.dtype, device=state.device)
