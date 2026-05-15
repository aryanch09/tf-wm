"""Tactile reconstruction decoder: latent → tactile tensor."""

from __future__ import annotations

import torch
from torch import nn


class TactileDecoder(nn.Module):
    """MLP decoder from latent space back to tactile measurements.

    Args:
        d_latent: Input latent dimension ``d_z``.
        n_taxels: Number of output taxels.
        n_channels: Number of channels per taxel.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.

    Shape:
        - Input  ``z``: ``(B, T, d_z)``
        - Output ``x_hat``: ``(B, T, N_taxel, C)``

    Example:
        >>> dec = TactileDecoder(d_latent=128, n_taxels=16, n_channels=1)
        >>> z = torch.randn(2, 10, 128)
        >>> dec(z).shape
        torch.Size([2, 10, 16, 1])
    """

    def __init__(
        self,
        d_latent: int = 128,
        n_taxels: int = 16,
        n_channels: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_taxels = n_taxels
        self.n_channels = n_channels
        layers: list[nn.Module] = [nn.Linear(d_latent, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, n_taxels * n_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to tactile.

        Args:
            z: Latent tensor ``(B, T, d_z)``.

        Returns:
            Reconstructed tactile ``(B, T, N_taxel, C)``.
        """
        B, T, _ = z.shape
        out = self.net(z)  # (B, T, N*C)
        return out.reshape(B, T, self.n_taxels, self.n_channels)
