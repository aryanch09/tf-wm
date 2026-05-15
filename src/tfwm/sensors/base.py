"""Base protocols and data containers for sensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from tfwm.types import Tactile


@dataclass(frozen=True, slots=True)
class SensorReading:
    """Timestamped sensor sample.

    Args:
        value: Sensor tensor value.
        timestamp_s: Monotonic timestamp in seconds.
    """

    value: torch.Tensor
    timestamp_s: float


@runtime_checkable
class TactileSensor(Protocol):
    """Protocol implemented by tactile sensors.

    Shape:
        ``read`` returns tactile data with shape ``(B, T, N_taxel, C)``.
    """

    sample_rate_hz: int
    n_taxels: int
    n_channels: int

    def read(self) -> Tactile:
        """Read the latest tactile sample."""
        ...

    def calibrate(self) -> None:
        """Calibrate or zero the sensor."""
        ...


class TactileSensorBase:
    """Abstract base class satisfying the TactileSensor Protocol.

    Subclasses must implement ``sample_rate_hz``, ``n_taxels``,
    ``n_channels``, ``read()``, and ``calibrate()``.
    """

    sample_rate_hz: int
    n_taxels: int
    n_channels: int

    def read(self) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError

    def calibrate(self) -> None:  # noqa: D102
        pass
