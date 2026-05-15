"""Cross-entropy method planner in latent space."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tfwm.errors import ShapeError
from tfwm.types import Action, TactileLatent


@dataclass(frozen=True, slots=True)
class CEMMPCConfig:
    """Configuration for CEM-MPC."""

    horizon: int = 20
    num_candidates: int = 256
    num_elites: int = 32
    num_iterations: int = 4
    action_dim: int = 6
    action_low: float = -1.0
    action_high: float = 1.0
    init_std: float = 0.5


class CEMMPCPlanner:
    """Latent-space MPC planner using the cross-entropy method."""

    def __init__(self, dynamics: torch.nn.Module, config: CEMMPCConfig | None = None) -> None:
        self.dynamics = dynamics
        self.config = config or CEMMPCConfig()
        if self.config.num_elites > self.config.num_candidates:
            raise ValueError("num_elites must be <= num_candidates")

    def _score(self, z_rollout: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        final = z_rollout[:, :, -1]
        while goal.dim() < final.dim():
            goal = goal.unsqueeze(1)
        return -torch.linalg.vector_norm(final - goal, dim=-1)

    @torch.no_grad()
    def plan(
        self,
        z_t: TactileLatent,
        goal: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> Action:
        """Return the first action from the best CEM sequence.

        Shape:
            ``z_t`` is ``(B, 1, d_z)`` or ``(B, d_z)``. Return is ``(B, d_a)``.
        """

        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
        if z_t.dim() != 3:
            raise ShapeError(f"expected z_t shape (B, T, d_z), got {tuple(z_t.shape)}")

        cfg = self.config
        batch = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype
        mean = torch.zeros(batch, cfg.horizon, cfg.action_dim, device=device, dtype=dtype)
        std = torch.full_like(mean, cfg.init_std)

        for _ in range(cfg.num_iterations):
            eps = torch.randn(
                batch, cfg.num_candidates, cfg.horizon, cfg.action_dim, device=device, dtype=dtype
            )
            candidates = torch.clamp(
                mean.unsqueeze(1) + std.unsqueeze(1) * eps, cfg.action_low, cfg.action_high
            )
            flat_actions = candidates.flatten(0, 1)
            z0 = z_t[:, -1:].repeat_interleave(cfg.num_candidates, dim=0)
            z_rollout, _ = self.dynamics.rollout(z0, flat_actions, context=context)
            scores = self._score(z_rollout.view(batch, cfg.num_candidates, cfg.horizon, -1), goal)
            elite_idx = scores.topk(cfg.num_elites, dim=1).indices
            gather_idx = elite_idx[..., None, None].expand(-1, -1, cfg.horizon, cfg.action_dim)
            elites = candidates.gather(1, gather_idx)
            mean = elites.mean(dim=1)
            std = elites.std(dim=1).clamp_min(1e-4)

        return mean[:, 0]
