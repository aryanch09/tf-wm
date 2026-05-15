"""Behaviour Cloning (BC) policy.

A simple imitation learning policy that regresses expert actions from the
current tactile latent.  Used in Phase 4 as a warm-start before RL fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tfwm.policies.base import PolicyBase


class BCPolicy(PolicyBase):
    """Behaviour cloning with a Gaussian action head.

    Args:
        d_latent: Input latent dimension.
        d_action: Output action dimension.
        hidden_dim: MLP hidden width.
        n_layers: Number of MLP layers.
        log_std_init: Initial log standard deviation.

    Shape:
        - ``z``: ``(B, d_z)`` — latest latent from encoder
        - Returns action: ``(B, d_a)``
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_action: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 3,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_latent, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, d_action)
        self.log_std = nn.Parameter(torch.full((d_action,), log_std_init))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_std) for the action distribution."""
        feat = self.trunk(z)
        return self.mu_head(feat), self.log_std.expand_as(self.mu_head(feat))

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Sample or take the mode of the policy.

        Args:
            obs: Dict with ``"latent"`` or ``"tactile_latent"`` key ``(B, d_z)``.
            deterministic: If ``True``, return the mean action.

        Returns:
            Action ``(B, d_a)``.
        """
        z = obs.get("latent") or obs.get("tactile_latent")
        if z is None:
            raise KeyError("obs must contain 'latent' or 'tactile_latent'")
        mu, log_std = self.forward(z)
        if deterministic:
            return mu
        std = log_std.exp()
        return mu + std * torch.randn_like(std)

    def loss(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood BC loss.

        Args:
            z: Latent states ``(B, d_z)``.
            actions: Expert actions ``(B, d_a)``.

        Returns:
            Scalar NLL loss.
        """
        mu, log_std = self.forward(z)
        std = log_std.exp()
        nll = 0.5 * ((actions - mu) / std).pow(2) + log_std + 0.5 * torch.log(torch.tensor(2 * 3.14159))
        return nll.mean()

    def reset(self) -> None:
        pass
