"""Binary cross-entropy loss for contact event classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ContactBCELoss(nn.Module):
    """Weighted cross-entropy over contact-event categories.

    Args:
        n_classes: Number of contact categories (none/stable/slip/edge/...).
        class_weights: Optional per-class weights to handle imbalance.
        label_smoothing: Label smoothing coefficient in ``[0, 1)``.

    Shape:
        - ``logits``:  ``(B, T, K)``
        - ``targets``: ``(B, T)`` integer class indices
        - Returns: scalar.
    """

    def __init__(
        self,
        n_classes: int = 5,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy contact event loss.

        Args:
            logits: Predicted logits ``(B, T, K)``.
            targets: Ground-truth class indices ``(B, T)``.

        Returns:
            Scalar cross-entropy loss.
        """
        B, T, K = logits.shape
        logits_flat = logits.reshape(B * T, K)
        targets_flat = targets.reshape(B * T).long()
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(
            logits_flat,
            targets_flat,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
