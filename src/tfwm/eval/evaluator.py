"""Evaluation metrics and routines."""

from __future__ import annotations

import torch
from torch import Tensor


class Evaluator:
    """Evaluation harness for world models."""

    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}

    def compute_mse(self, prediction: Tensor, target: Tensor) -> float:
        return float(torch.mean((prediction - target) ** 2).item())

    def compute_accuracy(self, logits: Tensor, labels: Tensor) -> float:
        predicted = torch.argmax(logits, dim=-1)
        return float((predicted == labels).float().mean().item())

    def add_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value

    def summary(self) -> dict[str, float]:
        return self.metrics.copy()
