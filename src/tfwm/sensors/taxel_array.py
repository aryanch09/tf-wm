"""In-memory taxel-array tactile sensor."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tfwm.errors import ShapeError
from tfwm.types import Tactile


@dataclass(frozen=True, slots=True)
class TaxelArraySensorConfig:
    """Configuration for a dense tactile taxel array."""

    sample_rate_hz: int = 1_000
    n_taxels: int = 64
    n_channels: int = 1
    baseline: float = 0.0


class TaxelArraySensor:
    """Simple concrete tactile sensor backed by a torch tensor buffer."""

    def __init__(self, config: TaxelArraySensorConfig | None = None) -> None:
        self.config = config or TaxelArraySensorConfig()
        self.sample_rate_hz = self.config.sample_rate_hz
        self.n_taxels = self.config.n_taxels
        self.n_channels = self.config.n_channels
        self._reading = torch.full(
            (1, 1, self.n_taxels, self.n_channels),
            float(self.config.baseline),
            dtype=torch.float32,
        )

    def set_reading(self, tactile: torch.Tensor) -> None:
        """Set the latest sensor value.

        Args:
            tactile: Tensor shaped ``(B, T, N_taxel, C)`` or ``(N_taxel, C)``.

        Raises:
            ShapeError: If the taxel/channel dimensions do not match the config.
        """

        if tactile.dim() == 2:
            tactile = tactile.unsqueeze(0).unsqueeze(0)
        if tactile.dim() != 4:
            raise ShapeError(f"expected tactile rank 4 or 2, got shape {tuple(tactile.shape)}")
        if tactile.shape[-2:] != (self.n_taxels, self.n_channels):
            raise ShapeError(
                "expected trailing tactile shape "
                f"{(self.n_taxels, self.n_channels)}, got {tuple(tactile.shape[-2:])}"
            )
        self._reading = tactile.detach().to(dtype=torch.float32).clone()

    def read(self) -> Tactile:
        """Return a cloned latest sample."""

        return self._reading.clone()

    def calibrate(self) -> None:
        """Reset the sensor to its configured baseline."""

        self._reading.fill_(float(self.config.baseline))
