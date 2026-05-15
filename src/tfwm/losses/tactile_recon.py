"""Tactile reconstruction loss (MSE + optional Laplacian smoothness)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TactileReconLoss(nn.Module):
    """Pixel-level MSE reconstruction with optional spatial smoothness.

    Args:
        reduction: ``"mean"`` or ``"sum"``.
        smoothness_weight: Weight for a finite-difference spatial smoothness
            penalty over the taxel grid.  Set to ``0`` (default) to disable.

    Shape:
        - ``pred``: ``(B, T, N_taxel, C)``
        - ``target``: same as ``pred``
        - Returns: scalar tensor.
    """

    def __init__(self, reduction: str = "mean", smoothness_weight: float = 0.0) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.reduction = reduction
        self.smoothness_weight = smoothness_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute MSE + smoothness loss.

        Args:
            pred: Predicted tactile ``(B, T, N, C)``.
            target: Ground-truth tactile ``(B, T, N, C)``.

        Returns:
            Scalar reconstruction loss.
        """
        mse = F.mse_loss(pred, target, reduction=self.reduction)
        if self.smoothness_weight == 0.0:
            return mse
        # Finite-difference along the taxel (spatial) dimension.
        diff = pred[:, :, 1:] - pred[:, :, :-1]
        smooth = diff.pow(2).mean() if self.reduction == "mean" else diff.pow(2).sum()
        return mse + self.smoothness_weight * smooth
