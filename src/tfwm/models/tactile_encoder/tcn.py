"""Causal temporal convolution tactile encoder."""

from __future__ import annotations

import torch
from torch import nn

from tfwm.errors import ShapeError
from tfwm.models.tactile_encoder.base import TactileEncoderBase
from tfwm.models.tactile_encoder.stochastic_head import StochasticHead
from tfwm.types import LatentDist, Tactile


class CausalConv1d(nn.Module):
    """One-dimensional causal convolution."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.left_pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution without future leakage."""

        return self.conv(torch.nn.functional.pad(x, (self.left_pad, 0)))


class TCNTactileEncoder(TactileEncoderBase):
    """Dilated TCN tactile encoder over flattened taxel channels."""

    def __init__(
        self,
        n_taxels: int,
        n_channels: int = 1,
        d_latent: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n_taxels = n_taxels
        self.n_channels = n_channels
        layers: list[nn.Module] = []
        in_dim = n_taxels * n_channels
        for idx in range(n_layers):
            layers.append(CausalConv1d(in_dim, hidden_dim, kernel_size, dilation=2**idx))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.head = StochasticHead(hidden_dim, d_latent)

    def forward(self, tactile: Tactile, attention_mask: torch.Tensor | None = None) -> LatentDist:
        """Encode tactile tensor into a latent distribution."""

        if tactile.dim() != 4:
            raise ShapeError(f"expected tactile shape (B, T, N, C), got {tuple(tactile.shape)}")
        if tactile.shape[-2:] != (self.n_taxels, self.n_channels):
            raise ShapeError(
                f"expected tactile trailing shape {(self.n_taxels, self.n_channels)}, "
                f"got {tuple(tactile.shape[-2:])}"
            )
        batch, time, _, _ = tactile.shape
        x = tactile.reshape(batch, time, -1).transpose(1, 2)
        features = self.network(x).transpose(1, 2)
        if attention_mask is not None:
            features = features * attention_mask.to(features.dtype).unsqueeze(-1)
        return self.head(features)
