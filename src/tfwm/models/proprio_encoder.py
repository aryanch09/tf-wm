"""Proprioception encoder: joint angles, velocities, torques → latent."""

from __future__ import annotations

import torch
from torch import nn


class ProprioEncoder(nn.Module):
    """MLP encoder for proprioceptive signals.

    Args:
        d_proprio: Input proprioception dimension.
        d_latent: Output embedding dimension.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        layer_norm: Apply LayerNorm before the output projection.

    Shape:
        - Input  ``proprio``: ``(B, T, d_p)``
        - Output ``embedding``: ``(B, T, d_latent)``
    """

    def __init__(
        self,
        d_proprio: int = 12,
        d_latent: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_proprio, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, d_latent))
        self.net = nn.Sequential(*layers)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """Encode proprioception sequence.

        Args:
            proprio: ``(B, T, d_p)``

        Returns:
            Embedding ``(B, T, d_latent)``.
        """
        return self.net(proprio)
