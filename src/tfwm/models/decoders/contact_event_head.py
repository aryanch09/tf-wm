"""Contact event classifier head: latent → contact category logits."""

from __future__ import annotations

import torch
from torch import nn


class ContactEventHead(nn.Module):
    """Classify contact events from latent state.

    Categories (default 5): none / stable / slip / edge / full_contact.

    Args:
        d_latent: Input latent dimension.
        n_classes: Number of contact-event categories.
        hidden_dim: Hidden layer width.

    Shape:
        - Input ``z``: ``(B, T, d_z)``
        - Output logits: ``(B, T, K)``
    """

    def __init__(self, d_latent: int = 128, n_classes: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return contact logits.

        Args:
            z: Latent tensor ``(B, T, d_z)``.

        Returns:
            Logits ``(B, T, K)``.
        """
        return self.net(z)
