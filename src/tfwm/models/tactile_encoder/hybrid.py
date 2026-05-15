"""Hybrid TCN→Transformer tactile encoder.

Local temporal patterns are captured by a stack of dilated causal convolutions
(TCN), then a causal transformer attends globally over the resulting feature
sequence.  This combination achieves sub-quadratic cost on long sequences
(the TCN down-samples the effective sequence density) while retaining the
long-range dependency modelling of attention.
"""

from __future__ import annotations

import torch
from torch import nn

from tfwm.errors import ShapeError
from tfwm.models.tactile_encoder.base import TactileEncoderBase
from tfwm.models.tactile_encoder.stochastic_head import StochasticHead
from tfwm.models.tactile_encoder.tcn import CausalConv1d
from tfwm.models.tactile_encoder.transformer import _CausalTransformerLayer, _SinusoidalPE
from tfwm.types import LatentDist


class HybridTactileEncoder(TactileEncoderBase):
    """TCN → Transformer causal tactile encoder.

    Stage 1 – TCN: ``n_tcn_layers`` dilated causal conv blocks extract local
    contact dynamics at exponentially growing receptive fields.

    Stage 2 – Transformer: ``n_tf_layers`` causal attention blocks integrate
    long-range context across the TCN feature sequence.

    Args:
        n_taxels: Number of taxels per timestep.
        n_channels: Channels per taxel.
        d_latent: Output latent size ``d_z``.
        tcn_hidden: TCN channel width (after input projection).
        n_tcn_layers: Number of dilated TCN blocks.
        tcn_kernel_size: Kernel size for each causal conv.
        d_model: Transformer hidden dimension (must equal ``tcn_hidden``).
        n_heads: Attention heads (must divide ``d_model``).
        n_tf_layers: Number of causal transformer layers.
        d_ff: Feed-forward expansion width.
        dropout: Dropout probability.

    Shape:
        - Input  ``tactile``: ``(B, T, N_taxel, C)``
        - Output ``mu`` / ``logvar``: ``(B, T, d_z)``

    Example:
        >>> enc = HybridTactileEncoder(n_taxels=16, n_channels=1, d_latent=128)
        >>> out = enc(torch.randn(2, 20, 16, 1))
        >>> out["mu"].shape
        torch.Size([2, 20, 128])
    """

    def __init__(
        self,
        n_taxels: int,
        n_channels: int = 1,
        d_latent: int = 128,
        tcn_hidden: int = 128,
        n_tcn_layers: int = 4,
        tcn_kernel_size: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_tf_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model != tcn_hidden:
            raise ValueError(
                f"d_model ({d_model}) must match tcn_hidden ({tcn_hidden}) so TCN output "
                "feeds directly into the transformer without an extra projection."
            )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.n_taxels = n_taxels
        self.n_channels = n_channels

        # ── Stage 1: TCN ──────────────────────────────────────────────
        tcn_layers: list[nn.Module] = []
        in_dim = n_taxels * n_channels
        for i in range(n_tcn_layers):
            tcn_layers.append(CausalConv1d(in_dim, tcn_hidden, tcn_kernel_size, dilation=2**i))
            tcn_layers.append(nn.GELU())
            tcn_layers.append(nn.Dropout(dropout))
            in_dim = tcn_hidden
        self.tcn = nn.Sequential(*tcn_layers)

        # ── Stage 2: Transformer ──────────────────────────────────────
        self.pos_enc = _SinusoidalPE(d_model)
        self.tf_layers = nn.ModuleList(
            [_CausalTransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_tf_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = StochasticHead(d_model, d_latent)

    def forward(self, tactile: torch.Tensor, attention_mask: torch.Tensor | None = None) -> LatentDist:
        """Encode tactile stream via TCN then causal transformer.

        Args:
            tactile: ``(B, T, N_taxel, C)``
            attention_mask: Optional ``(B, T)`` boolean mask; ``True`` = valid.

        Returns:
            :class:`~tfwm.types.LatentDist` with ``mu``, ``logvar`` ``(B, T, d_z)``.

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

        # TCN operates on (B, in_channels, T) — channels-first convention.
        x = tactile.reshape(B, T, N * C).transpose(1, 2)  # (B, N*C, T)
        x = self.tcn(x).transpose(1, 2)  # (B, T, tcn_hidden)

        # Mask out padded positions before transformer.
        if attention_mask is not None:
            x = x * attention_mask.to(x.dtype).unsqueeze(-1)

        # Transformer stage.
        x = self.pos_enc(x)
        key_padding_mask = (~attention_mask.bool()) if attention_mask is not None else None
        for layer in self.tf_layers:
            x = layer(x, key_padding_mask)

        x = self.norm(x)
        return self.head(x)
