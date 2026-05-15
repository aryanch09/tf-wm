"""Base gate contract."""

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from tfwm.types import GateScore, TactileLatent


class Gate(nn.Module):
    """Base class for deciding when vision should be queried."""

    threshold: float

    @abstractmethod
    def forward(self, latent: TactileLatent, dynamics_aux: dict[str, torch.Tensor]) -> GateScore:
        """Return query scores in ``[0, 1]`` with shape ``(B, T)``."""

    def query_vision(self, score: GateScore) -> torch.Tensor:
        """Convert scores to boolean query decisions."""

        return score > self.threshold
