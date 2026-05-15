"""Gated cross-attention context fusion.

Fuses a primary tactile latent sequence with optional vision and proprioception
embeddings using a learned soft gate: the gate score controls how much
context is blended into the tactile stream.  When vision is unavailable the
gate collapses to identity — matching the VOI-gating philosophy of TF-WM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GatedContextFusion(nn.Module):
    """Cross-attention fusion with learned gating for vision/proprio context.

    Architecture:
        1. Multi-head cross-attention from tactile latent (query) to
           context (key/value).
        2. A scalar gate ``g ∈ [0,1]`` modulates the residual:
           ``out = tactile + g * attn_out``.
        3. The gate is conditioned on both the tactile latent and the
           attended context to make zeroing-out of missing modalities smooth.

    Args:
        d_tactile: Tactile latent dimension.
        d_context: Context embedding dimension (vision / proprio).
        n_heads: Attention heads (must divide ``d_tactile``).
        dropout: Attention dropout.

    Shape:
        - ``tactile``: ``(B, T, d_tactile)``
        - ``context``: ``(B, T_ctx, d_context)`` or ``None``
        - Returns ``(B, T, d_tactile)``
    """

    def __init__(
        self,
        d_tactile: int = 128,
        d_context: int = 256,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_tactile % n_heads != 0:
            raise ValueError(f"d_tactile={d_tactile} must be divisible by n_heads={n_heads}")
        self.context_proj = nn.Linear(d_context, d_tactile)
        self.attn = nn.MultiheadAttention(d_tactile, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_tactile)
        # Gate: conditioned on [tactile, attn_out]
        self.gate = nn.Sequential(
            nn.Linear(d_tactile * 2, d_tactile),
            nn.SiLU(),
            nn.Linear(d_tactile, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tactile: torch.Tensor,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse tactile with optional context via gated cross-attention.

        Args:
            tactile: Primary tactile latent ``(B, T, d_tactile)``.
            context: Context embedding ``(B, T_ctx, d_context)`` or ``None``.
                When ``None`` the module returns ``tactile`` unchanged.
            context_mask: Boolean key-padding mask for context ``(B, T_ctx)``.
                ``True`` = padding (ignored).

        Returns:
            Fused tactile latent ``(B, T, d_tactile)``.
        """
        if context is None:
            return tactile

        ctx = self.context_proj(context)  # (B, T_ctx, d_tactile)
        attn_out, _ = self.attn(
            self.norm(tactile), ctx, ctx,
            key_padding_mask=context_mask,
            need_weights=False,
        )
        g = self.gate(torch.cat([tactile, attn_out], dim=-1))  # (B, T, 1)
        return tactile + g * attn_out


class MultiModalFusion(nn.Module):
    """Sequential fusion of vision and proprioception into the tactile stream.

    Applies :class:`GatedContextFusion` twice — first for vision, then for
    proprioception — so each modality has independent gating.

    Args:
        d_tactile: Tactile latent dimension.
        d_vision: Vision embedding dimension.
        d_proprio: Proprioception embedding dimension.
        n_heads: Attention heads.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        d_tactile: int = 128,
        d_vision: int = 512,
        d_proprio: int = 64,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vision_fusion = GatedContextFusion(d_tactile, d_vision, n_heads, dropout)
        self.proprio_fusion = GatedContextFusion(d_tactile, d_proprio, n_heads, dropout)

    def forward(
        self,
        tactile: torch.Tensor,
        vision: torch.Tensor | None = None,
        proprio: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        proprio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse all available modalities into the tactile latent.

        Args:
            tactile: ``(B, T, d_tactile)``
            vision:  ``(B, T_v, d_vision)`` or ``None``.
            proprio: ``(B, T, d_proprio)`` or ``None``.
            vision_mask: Key-padding mask for vision.
            proprio_mask: Key-padding mask for proprio.

        Returns:
            Fused tactile latent ``(B, T, d_tactile)``.
        """
        x = self.vision_fusion(tactile, vision, vision_mask)
        x = self.proprio_fusion(x, proprio, proprio_mask)
        return x
