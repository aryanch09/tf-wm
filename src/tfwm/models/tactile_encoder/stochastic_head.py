"""Stochastic latent head for tactile encoders."""

from __future__ import annotations

import torch
from torch import nn

from tfwm.types import LatentDist


class StochasticHead(nn.Module):
    """Project features to Gaussian latent parameters."""

    def __init__(self, in_dim: int, d_latent: int = 128) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, d_latent)
        self.logvar = nn.Linear(in_dim, d_latent)

    def forward(self, x: torch.Tensor) -> LatentDist:
        """Return ``{"mu", "logvar"}`` for the final feature dimension."""

        return {"mu": self.mu(x), "logvar": self.logvar(x).clamp(-10.0, 5.0)}
