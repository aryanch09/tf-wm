"""ReSkin barometric tactile sensor driver."""

from __future__ import annotations

import numpy as np
import torch

from tfwm.sensors.base import TactileSensorBase


class ReSkinSensor(TactileSensorBase):
    """ReSkin magnetic tactile sensor.

    ReSkin uses an array of magnetometers to sense contact via deformation of
    a magnetic elastomer.  Each magnetometer reports (Bx, By, Bz) → 3-channel
    taxel readings after baseline subtraction.

    Args:
        n_taxels: Number of magnetometer elements.
        sample_rate_hz: Acquisition rate (up to ~400 Hz for ReSkin).
        port: Serial port string (``"/dev/ttyACM0"`` on Linux).
    """

    def __init__(
        self,
        n_taxels: int = 5,
        sample_rate_hz: int = 400,
        port: str = "/dev/ttyACM0",
    ) -> None:
        self._n_taxels = n_taxels
        self._sample_rate_hz = sample_rate_hz
        self._n_channels = 3  # Bx, By, Bz
        self._port = port
        self._baseline: np.ndarray | None = None

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def n_taxels(self) -> int:
        return self._n_taxels

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def calibrate(self) -> None:
        """Record the no-contact magnetic baseline."""
        samples = [self._read_raw() for _ in range(50)]
        self._baseline = np.mean(samples, axis=0)

    def read(self) -> torch.Tensor:
        """Return one frame of baseline-subtracted readings.

        Returns:
            Tensor ``(1, 1, n_taxels, 3)`` float32.
        """
        raw = self._read_raw()  # (n_taxels, 3)
        if self._baseline is not None:
            raw = raw - self._baseline
        # Normalise to rough [-1, 1] using ±50 μT range.
        raw = np.clip(raw / 50.0, -1.0, 1.0).astype(np.float32)
        return torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)

    def _read_raw(self) -> np.ndarray:
        """Stub: read (n_taxels, 3) magnetic field values in μT."""
        return np.random.normal(0, 5.0, (self._n_taxels, 3))
