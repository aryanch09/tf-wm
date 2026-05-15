"""Base tactile encoder contract."""

from __future__ import annotations

from abc import abstractmethod

from torch import nn

from tfwm.types import LatentDist, Tactile


class TactileEncoderBase(nn.Module):
    """Contract-compatible tactile encoder.

    Inputs:
        tactile: ``(B, T, N_taxel, C)``.

    Returns:
        Latent distribution with ``mu`` and ``logvar`` shaped ``(B, T, d_z)``.
    """

    @abstractmethod
    def forward(self, tactile: Tactile, attention_mask=None) -> LatentDist:
        """Encode tactile streams."""
