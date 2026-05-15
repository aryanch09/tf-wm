"""Transformer-based causal tactile encoder.

Implements a causal (masked) multi-head self-attention encoder over the
taxel-channel feature sequence.  Causality is enforced via an additive
upper-triangular mask so the encoder can be used autoregressively during
deployment without recomputation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from tfwm.errors import ShapeError
from tfwm.models.tactile_encoder.base import TactileEncoderBase
from tfwm.models.tactile_encoder.stochastic_head import StochasticHead
from tfwm.types import LatentDist


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding; cached up to ``max_len``."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


class _CausalTransformerLayer(nn.Module):
    """Single pre-norm causal transformer layer with rotary-free ALiBi masking."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular additive mask (−∞ for future positions)."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask.triu_(diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = x.size(1)
        causal = self._causal_mask(T, x.device)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal,
                                key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerTactileEncoder(TactileEncoderBase):
    """Causal transformer encoder for tactile streams.

    Flattens the ``(N_taxel, C)`` spatial dimension with a learned linear
    projection, adds sinusoidal positional encodings, then applies ``n_layers``
    causal transformer blocks.

    Args:
        n_taxels: Number of taxels per timestep.
        n_channels: Channels per taxel.
        d_latent: Output latent size ``d_z``.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads (must divide ``d_model``).
        n_layers: Number of transformer layers.
        d_ff: Feed-forward expansion dimension.
        dropout: Dropout probability applied inside attention and FF.

    Shape:
        - Input ``tactile``: ``(B, T, N_taxel, C)``
        - Output ``mu`` / ``logvar``: ``(B, T, d_z)``

    Example:
        >>> enc = TransformerTactileEncoder(n_taxels=16, n_channels=1, d_latent=128)
        >>> out = enc(torch.randn(2, 10, 16, 1))
        >>> out["mu"].shape
        torch.Size([2, 10, 128])
    """

    def __init__(
        self,
        n_taxels: int,
        n_channels: int = 1,
        d_latent: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_taxels = n_taxels
        self.n_channels = n_channels
        self.input_proj = nn.Linear(n_taxels * n_channels, d_model)
        self.pos_enc = _SinusoidalPE(d_model)
        self.layers = nn.ModuleList(
            [_CausalTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = StochasticHead(d_model, d_latent)

    def forward(self, tactile: torch.Tensor, attention_mask: torch.Tensor | None = None) -> LatentDist:
        """Encode tactile stream causally.

        Args:
            tactile: ``(B, T, N_taxel, C)``
            attention_mask: Optional boolean mask ``(B, T)``; ``True`` means
                padding (ignored position).

        Returns:
            :class:`~tfwm.types.LatentDist` with shapes ``(B, T, d_z)``.

        Raises:
            ShapeError: On shape mismatch.
        """
        if tactile.dim() != 4:
            raise ShapeError(f"Expected tactile (B, T, N, C), got {tuple(tactile.shape)}")
        B, T, N, C = tactile.shape
        if (N, C) != (self.n_taxels, self.n_channels):
            raise ShapeError(
                f"Expected N={self.n_taxels}, C={self.n_channels}; got N={N}, C={C}"
            )
        x = tactile.reshape(B, T, N * C)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        key_padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # attention_mask: True=valid → key_padding_mask: True=ignore
            key_padding_mask = ~attention_mask.bool()

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)
        return self.head(x)
