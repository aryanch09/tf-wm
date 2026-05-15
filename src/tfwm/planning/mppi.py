"""Model Predictive Path Integral (MPPI) planner in latent space.

MPPI samples action trajectories from a Gaussian importance distribution and
computes a soft-optimal weighted update in a single forward pass — no
elite-selection loop required.  This makes it faster than CEM for smooth
cost landscapes while offering lower variance than random shooting.

References:
    Williams et al. (2017) "Information Theoretic MPC for Model-Based
    Reinforcement Learning." ICRA 2017.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tfwm.errors import ShapeError


@dataclass(frozen=True, slots=True)
class MPPIConfig:
    """Configuration for MPPI."""

    horizon: int = 20
    num_samples: int = 512
    action_dim: int = 6
    action_low: float = -1.0
    action_high: float = 1.0
    noise_std: float = 0.5
    temperature: float = 1.0       # λ in the MPPI objective
    num_iterations: int = 1        # MPPI typically runs once; >1 refines the mean


class MPPIPlanner:
    """Latent-space MPPI planner.

    Args:
        dynamics: Module with a :meth:`rollout` method compatible with the
            :class:`~tfwm.types.Dynamics` Protocol.
        config: :class:`MPPIConfig` parameters.

    Shape:
        - ``z_t``: ``(B, 1, d_z)`` or ``(B, d_z)``
        - Returns action: ``(B, d_a)``
    """

    def __init__(self, dynamics: torch.nn.Module, config: MPPIConfig | None = None) -> None:
        self.dynamics = dynamics
        self.config = config or MPPIConfig()

    def _cost(self, z_rollout: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity to goal at the final horizon step.

        Args:
            z_rollout: ``(B, K, H, d_z)``
            goal: ``(B, d_z)``

        Returns:
            Cost tensor ``(B, K)``; lower is better.
        """
        final = z_rollout[:, :, -1]  # (B, K, d_z)
        g = goal.unsqueeze(1)        # (B, 1, d_z)
        return torch.linalg.vector_norm(final - g, dim=-1)

    @torch.no_grad()
    def plan(
        self,
        z_t: torch.Tensor,
        goal: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the first action from the MPPI-optimal trajectory.

        Args:
            z_t: Current latent state ``(B, 1, d_z)`` or ``(B, d_z)``.
            goal: Goal latent ``(B, d_z)``.
            context: Optional context passed to dynamics rollout.

        Returns:
            First-step action ``(B, d_a)``.

        Raises:
            ShapeError: If ``z_t`` has wrong rank.
        """
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
        if z_t.dim() != 3:
            raise ShapeError(f"Expected z_t (B, T, d_z), got {tuple(z_t.shape)}")

        cfg = self.config
        B = z_t.shape[0]
        K = cfg.num_samples
        H = cfg.horizon
        D = cfg.action_dim
        device, dtype = z_t.device, z_t.dtype

        mean = torch.zeros(B, H, D, device=device, dtype=dtype)

        for _ in range(cfg.num_iterations):
            noise = torch.randn(B, K, H, D, device=device, dtype=dtype) * cfg.noise_std
            perturbed = (mean.unsqueeze(1) + noise).clamp(cfg.action_low, cfg.action_high)

            flat_actions = perturbed.flatten(0, 1)  # (B*K, H, D)
            z0 = z_t[:, -1:].repeat_interleave(K, dim=0)
            z_pred, _ = self.dynamics.rollout(z0, flat_actions, context=context)
            z_pred = z_pred.view(B, K, H, -1)

            costs = self._cost(z_pred, goal)  # (B, K)
            weights = torch.softmax(-costs / cfg.temperature, dim=1)  # (B, K)
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * perturbed).sum(dim=1)

        return mean[:, 0].clamp(cfg.action_low, cfg.action_high)
