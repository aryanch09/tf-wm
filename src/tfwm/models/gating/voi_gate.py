"""Learned value-of-information gate."""

from __future__ import annotations

import torch
from torch import nn

from tfwm.errors import ShapeError
from tfwm.models.gating.base import Gate
from tfwm.types import GateScore, TactileLatent


class VOIGate(Gate):
    """Small MLP that predicts the value of querying vision."""

    def __init__(self, d_latent: int = 128, hidden_dim: int = 128, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.network = nn.Sequential(
            nn.Linear(d_latent * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: TactileLatent, dynamics_aux: dict[str, torch.Tensor]) -> GateScore:
        """Predict VOI scores from latent and predicted uncertainty."""

        if latent.dim() != 3:
            raise ShapeError(f"expected latent shape (B, T, d_z), got {tuple(latent.shape)}")
        logvar = dynamics_aux.get("logvar", torch.zeros_like(latent))
        if logvar.shape != latent.shape:
            raise ShapeError(
                f"expected logvar shape {tuple(latent.shape)}, got {tuple(logvar.shape)}"
            )
        return torch.sigmoid(self.network(torch.cat([latent, logvar], dim=-1))).squeeze(-1)
