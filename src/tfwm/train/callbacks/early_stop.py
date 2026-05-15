"""Early stopping callback based on a validation metric."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class EarlyStopCallback:
    """Stop training when a metric stops improving.

    Args:
        model: Model to checkpoint at best metric.
        patience: Number of evaluations without improvement before stopping.
        min_delta: Minimum absolute improvement to count as improvement.
        mode: ``"min"`` (lower is better) or ``"max"``.
        save_path: Where to save the best checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        save_path: Path = Path("outputs/best.pt"),
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Record a new metric value; returns ``True`` if training should stop.

        Args:
            metric: The current validation metric.

        Returns:
            ``True`` when the patience budget is exhausted.
        """
        improved = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
