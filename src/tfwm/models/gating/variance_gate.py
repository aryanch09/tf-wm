"""Uncertainty-driven vision gate."""

from __future__ import annotations

import torch

from tfwm.errors import ShapeError
from tfwm.models.gating.base import Gate
from tfwm.types import GateScore, TactileLatent


class VarianceGate(Gate):
    """Monotone gate over predicted latent log-variance."""

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = max(float(temperature), 1e-6)
        self.bias = float(bias)

    def forward(self, latent: TactileLatent, dynamics_aux: dict[str, torch.Tensor]) -> GateScore:
        """Score uncertainty from ``dynamics_aux["logvar"]``.

        Raises:
            ShapeError: If the log-variance is absent or not rank 3.
        """

        del latent
        logvar = dynamics_aux.get("logvar")
        if logvar is None:
            raise ShapeError('VarianceGate requires dynamics_aux["logvar"].')
        if logvar.dim() != 3:
            raise ShapeError(f"expected logvar shape (B, T, d_z), got {tuple(logvar.shape)}")
        uncertainty = logvar.exp().mean(dim=-1)
        return torch.sigmoid((uncertainty + self.bias) / self.temperature)
