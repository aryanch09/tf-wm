"""Tactile sensor calibration utilities.

Provides per-taxel baseline estimation, drift correction, and a noise model
characterisation routine.  Results are serialised to YAML for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class TactileCalibrator:
    """Collect and persist per-taxel calibration coefficients.

    Args:
        n_taxels: Number of taxels.
        n_channels: Channels per taxel.
        n_frames: Number of no-contact frames to average for baseline.

    Usage::

        cal = TactileCalibrator(n_taxels=16)
        cal.collect_baseline(sensor)
        cal.save("calibration.yaml")
    """

    def __init__(self, n_taxels: int, n_channels: int = 1, n_frames: int = 200) -> None:
        self.n_taxels = n_taxels
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.baseline: np.ndarray | None = None
        self.gain: np.ndarray | None = None
        self.noise_std: float = 0.0

    def collect_baseline(self, sensor: object) -> None:
        """Record rest-state baseline.

        Args:
            sensor: Object with a ``read()`` method returning
                ``(1, 1, n_taxels, n_channels)`` tensors.
        """
        import torch
        readings: list[np.ndarray] = []
        for _ in range(self.n_frames):
            t = sensor.read()  # type: ignore[attr-defined]
            readings.append(t.squeeze().numpy())
        stack = np.stack(readings, axis=0)  # (n_frames, n_taxels, n_channels)
        self.baseline = stack.mean(axis=0)
        self.noise_std = float(stack.std())

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """Subtract baseline and apply gain.

        Args:
            raw: Raw taxel reading ``(n_taxels, n_channels)``.

        Returns:
            Calibrated reading ``(n_taxels, n_channels)`` clipped to ``[0, 1]``.
        """
        if self.baseline is not None:
            raw = raw - self.baseline
        if self.gain is not None:
            raw = raw * self.gain
        return np.clip(raw, 0.0, 1.0)

    def save(self, path: Path | str) -> None:
        """Serialise calibration to YAML."""
        data: dict[str, object] = {
            "n_taxels": self.n_taxels,
            "n_channels": self.n_channels,
            "noise_std": float(self.noise_std),
            "baseline": self.baseline.tolist() if self.baseline is not None else None,
            "gain": self.gain.tolist() if self.gain is not None else None,
        }
        with open(path, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "TactileCalibrator":
        """Load calibration from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        cal = cls(n_taxels=data["n_taxels"], n_channels=data["n_channels"])
        cal.noise_std = data["noise_std"]
        if data["baseline"] is not None:
            cal.baseline = np.array(data["baseline"], dtype=np.float32)
        if data["gain"] is not None:
            cal.gain = np.array(data["gain"], dtype=np.float32)
        return cal
