"""Affordance and object-state prediction losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AffordanceLoss(nn.Module):
    """Cross-entropy over discrete affordance logits.

    Args:
        n_affordances: Number of affordance classes.
        label_smoothing: Label smoothing in ``[0, 1)``.

    Shape:
        - ``logits``:  ``(B, T, K)`` or ``(B, K)``
        - ``targets``: ``(B, T)`` or ``(B,)`` integer indices
        - Returns: scalar.
    """

    def __init__(self, n_affordances: int = 8, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.n_affordances = n_affordances
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute affordance cross-entropy.

        Args:
            logits: Predicted logits of shape ``(..., K)``.
            targets: Integer ground-truth indices of shape ``(...)``.

        Returns:
            Scalar cross-entropy loss.
        """
        return F.cross_entropy(
            logits.reshape(-1, self.n_affordances),
            targets.reshape(-1).long(),
            label_smoothing=self.label_smoothing,
        )


class ObjectStateLoss(nn.Module):
    """Regression loss for continuous object state (pose, velocity).

    Args:
        reduction: ``"mean"`` or ``"sum"``.

    Shape:
        - ``pred``:   ``(B, T, d_state)``
        - ``target``: ``(B, T, d_state)``
        - Returns: scalar.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 regression over predicted object states.

        Args:
            pred: Predicted state ``(B, T, d)``.
            target: Ground-truth state ``(B, T, d)``.

        Returns:
            Scalar smooth-L1 loss.
        """
        return F.smooth_l1_loss(pred, target, reduction=self.reduction)
