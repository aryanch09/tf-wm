"""Standard loss functions for TF-WM."""

from __future__ import annotations

import torch
from torch import Tensor


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2)


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
