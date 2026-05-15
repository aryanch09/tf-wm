"""Soft Actor-Critic (SAC) policy for Phase 4 RL fine-tuning.

Implements a standard SAC actor (reparameterised Gaussian with tanh squashing)
and twin-critic (clipped double-Q) in latent space.

References:
    Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy
    Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tfwm.policies.base import PolicyBase

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SACPolicy(PolicyBase):
    """SAC actor: latent → tanh-squashed Gaussian action.

    Args:
        d_latent: Observation (latent) dimension.
        d_action: Action dimension.
        action_scale: Scales the tanh output to ``[-action_scale, action_scale]``.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_action: int = 6,
        action_scale: float = 1.0,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.action_scale = action_scale
        self.trunk = _mlp(d_latent, hidden_dim, hidden_dim, n_layers - 1)
        self.mu_head = nn.Linear(hidden_dim, d_action)
        self.log_std_head = nn.Linear(hidden_dim, d_action)

    def _dist(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(z)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mu, log_std

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Sample action (or mode) from the actor.

        Args:
            obs: Dict with ``"latent"`` or ``"tactile_latent"`` key ``(B, d_z)``.
            deterministic: Return tanh(mu) without noise.

        Returns:
            Action ``(B, d_a)`` in ``[-action_scale, action_scale]``.
        """
        z = obs.get("latent") or obs.get("tactile_latent")
        if z is None:
            raise KeyError("obs must contain 'latent' or 'tactile_latent'")
        mu, log_std = self._dist(z)
        if deterministic:
            return torch.tanh(mu) * self.action_scale
        std = log_std.exp()
        x = mu + std * torch.randn_like(std)
        return torch.tanh(x) * self.action_scale

    def log_prob(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute its log-probability.

        Args:
            z: Latent state ``(B, d_z)``.

        Returns:
            ``(action, log_prob)`` tensors of shape ``(B, d_a)`` and ``(B,)``.
        """
        mu, log_std = self._dist(z)
        std = log_std.exp()
        x = mu + std * torch.randn_like(std)
        action = torch.tanh(x) * self.action_scale
        # Log-prob under tanh-Gaussian (with change-of-variables correction).
        log_p = (-0.5 * ((x - mu) / std.clamp_min(1e-8)).pow(2) - log_std
                 - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_p -= (2.0 * (math.log(2) - x - F.softplus(-2 * x))).sum(-1)
        return action, log_p

    def reset(self) -> None:
        pass


class TwinCritic(nn.Module):
    """Clipped double-Q critic for SAC.

    Args:
        d_latent: Observation (latent) dimension.
        d_action: Action dimension.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_action: int = 6,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.q1 = _mlp(d_latent + d_action, hidden_dim, 1, n_layers)
        self.q2 = _mlp(d_latent + d_action, hidden_dim, 1, n_layers)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q-values.

        Args:
            z: Latent state ``(B, d_z)``.
            action: Action ``(B, d_a)``.

        Returns:
            ``(Q1, Q2)`` each of shape ``(B, 1)``.
        """
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return ``min(Q1, Q2)`` for target computation."""
        q1, q2 = self.forward(z, action)
        return torch.min(q1, q2)
