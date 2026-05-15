"""Gradient clipping callback with optional per-group tracking."""

from __future__ import annotations

import torch
from torch import nn


class GradClipCallback:
    """Clip gradients and optionally log the per-step norm.

    Args:
        model: Module whose parameters are clipped.
        max_norm: Maximum gradient L2 norm.
        norm_type: ``p`` in the Lp norm (default 2.0).
        log_norm: Whether to return the pre-clip norm in :meth:`step`.
    """

    def __init__(
        self, model: nn.Module, max_norm: float = 10.0, norm_type: float = 2.0, log_norm: bool = True
    ) -> None:
        self.model = model
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.log_norm = log_norm

    def step(self) -> float | None:
        """Clip gradients in-place; return pre-clip norm if ``log_norm``."""
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return None
        total_norm = float(
            torch.linalg.vector_norm(
                torch.stack([p.grad.detach().norm(self.norm_type) for p in params]),
                ord=self.norm_type,
            ).item()
        )
        torch.nn.utils.clip_grad_norm_(params, self.max_norm, norm_type=self.norm_type)
        return total_norm if self.log_norm else None
