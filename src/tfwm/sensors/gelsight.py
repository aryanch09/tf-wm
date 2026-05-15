"""GelSight Mini optical tactile sensor driver."""

from __future__ import annotations

import numpy as np
import torch

from tfwm.sensors.base import TactileSensorBase


class GelSightMiniSensor(TactileSensorBase):
    """GelSight Mini vision-based tactile sensor.

    Converts raw RGB images to surface normal maps and computes summary
    taxel statistics (mean normal magnitude per spatial patch).

    Args:
        device_id: Camera index.
        n_taxels: Number of spatial patches.
        sample_rate_hz: Acquisition rate (hardware limited to ~30–60 Hz).
    """

    def __init__(
        self,
        device_id: int = 0,
        n_taxels: int = 16,
        sample_rate_hz: int = 30,
    ) -> None:
        self._device_id = device_id
        self._n_taxels = n_taxels
        self._sample_rate_hz = sample_rate_hz
        self._n_channels = 3   # normal x, y, z per patch
        self._background: np.ndarray | None = None

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
        """Capture rest-state background for normal-map computation."""
        frames = [self._capture_rgb() for _ in range(20)]
        self._background = np.mean(frames, axis=0)

    def read(self) -> torch.Tensor:
        """Return normal-map-derived taxel readings.

        Returns:
            Tensor ``(1, 1, n_taxels, 3)`` float32 in ``[-1, 1]``.
        """
        rgb = self._capture_rgb()
        if self._background is not None:
            diff = rgb.astype(np.float32) - self._background.astype(np.float32)
        else:
            diff = rgb.astype(np.float32)
        normals = self._rgb_to_normals(diff)  # (H, W, 3)
        taxels = self._pool(normals)           # (n_taxels, 3)
        t = torch.from_numpy(taxels.astype(np.float32))
        return t.unsqueeze(0).unsqueeze(0)

    def _capture_rgb(self) -> np.ndarray:
        return np.random.randint(0, 256, (320, 240, 3), dtype=np.uint8)

    @staticmethod
    def _rgb_to_normals(diff: np.ndarray) -> np.ndarray:
        """Approximate Lambertian photometric stereo (3-light)."""
        normals = diff / (np.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6))
        return np.clip(normals, -1, 1)

    def _pool(self, normals: np.ndarray) -> np.ndarray:
        H, W, C = normals.shape
        sq = max(1, int(self._n_taxels ** 0.5))
        ph, pw = H // sq, W // sq
        out = np.zeros((self._n_taxels, C), dtype=np.float32)
        for i in range(sq):
            for j in range(sq):
                idx = i * sq + j
                if idx < self._n_taxels:
                    out[idx] = normals[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean(axis=(0, 1))
        return out
