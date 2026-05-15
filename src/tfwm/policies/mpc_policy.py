"""MPC policy: encode observation → plan with CEM → execute first action."""

from __future__ import annotations

import torch

from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.planning.cem_mpc import CEMMPCPlanner
from tfwm.planning.safety_filter import SafetyFilter
from tfwm.policies.base import PolicyBase


class MPCPolicy(PolicyBase):
    """Latent-space MPC policy using CEM.

    Args:
        encoder: Tactile encoder to obtain ``z_t``.
        planner: CEM-MPC planner.
        goal: Fixed task goal latent ``(d_z,)`` or ``(1, d_z)``.
        safety_filter: Optional safety filter (wraps actions).

    Shape:
        - ``obs["tactile"]``: ``(B, T_window, N_taxel, C)``
        - Returns action: ``(B, d_a)``
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        planner: CEMMPCPlanner,
        goal: torch.Tensor,
        safety_filter: SafetyFilter | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.planner = planner
        self.safety_filter = safety_filter
        self.register_buffer("goal", goal)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """Encode current tactile observation and plan.

        Args:
            obs: Dict with ``"tactile"`` key ``(B, T, N, C)``.
            deterministic: Unused (MPC is deterministic by construction).

        Returns:
            Action ``(B, d_a)``.
        """
        del deterministic
        tactile = obs["tactile"]
        enc = self.encoder(tactile)
        z_t = enc["mu"]  # (B, T, d_z)
        B = z_t.shape[0]
        goal = self.goal.expand(B, -1)
        action = self.planner.plan(z_t, goal)
        if self.safety_filter is not None:
            action = self.safety_filter(action, z_t[:, -1])
        return action

    def reset(self) -> None:
        pass
