"""Affordance prediction head: latent → affordance logits."""

from __future__ import annotations

import torch
from torch import nn


class AffordanceHead(nn.Module):
    """Multi-label affordance classifier from latent state.

    Args:
        d_latent: Input latent dimension.
        n_affordances: Number of affordance classes.
        hidden_dim: Hidden layer width.
        multilabel: If ``True``, outputs raw logits for sigmoid (multi-label);
            otherwise outputs for softmax (single-label).

    Shape:
        - Input  ``z``: ``(B, T, d_z)``
        - Output logits: ``(B, T, K)``
    """

    def __init__(
        self,
        d_latent: int = 128,
        n_affordances: int = 8,
        hidden_dim: int = 128,
        multilabel: bool = False,
    ) -> None:
        super().__init__()
        self.multilabel = multilabel
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_affordances),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return affordance logits.

        Args:
            z: Latent tensor ``(B, T, d_z)`` or ``(B, d_z)``.

        Returns:
            Logits of shape ``(B, T, K)`` or ``(B, K)``.
        """
        return self.net(z)
