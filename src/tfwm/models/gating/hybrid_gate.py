"""Hybrid uncertainty and learned-VOI gate."""

from __future__ import annotations

import torch

from tfwm.models.gating.base import Gate
from tfwm.models.gating.variance_gate import VarianceGate
from tfwm.models.gating.voi_gate import VOIGate
from tfwm.types import GateScore, TactileLatent


class HybridGate(Gate):
    """Blend a monotone variance gate with a learned VOI gate."""

    def __init__(
        self,
        d_latent: int = 128,
        threshold: float = 0.5,
        variance_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.variance_weight = float(variance_weight)
        self.variance_gate = VarianceGate(threshold=threshold)
        self.voi_gate = VOIGate(d_latent=d_latent, threshold=threshold)

    def forward(self, latent: TactileLatent, dynamics_aux: dict[str, torch.Tensor]) -> GateScore:
        """Return a convex blend of variance and learned-VOI scores."""

        w = min(max(self.variance_weight, 0.0), 1.0)
        return w * self.variance_gate(latent, dynamics_aux) + (1.0 - w) * self.voi_gate(
            latent, dynamics_aux
        )
