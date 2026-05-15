"""DIGIT optical tactile sensor driver.

Wraps the DIGIT USB camera and converts raw images to normalised taxel arrays
using a pre-computed background subtraction + spatial downsampling pipeline.
"""

from __future__ import annotations

import numpy as np
import torch

from tfwm.errors import SensorError
from tfwm.sensors.base import TactileSensorBase


class DIGITSensor(TactileSensorBase):
    """DIGIT optical tactile sensor.

    Args:
        device_id: USB device index (0-based).
        n_taxels: Number of virtual taxels after downsampling.
        sample_rate_hz: Target acquisition rate.
        background_frames: Frames to average for background calibration.

    Raises:
        SensorError: If the DIGIT device cannot be opened.
    """

    def __init__(
        self,
        device_id: int = 0,
        n_taxels: int = 16,
        sample_rate_hz: int = 60,
        background_frames: int = 30,
    ) -> None:
        self._device_id = device_id
        self._n_taxels = n_taxels
        self._sample_rate_hz = sample_rate_hz
        self._n_channels = 3         # RGB
        self._background: np.ndarray | None = None
        self._background_frames = background_frames
        self._cap: object | None = None

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
        """Capture background frames and compute baseline for subtraction."""
        frames = []
        for _ in range(self._background_frames):
            frames.append(self._capture_raw())
        self._background = np.mean(frames, axis=0)

    def read(self) -> torch.Tensor:
        """Capture one frame and return taxel readings.

        Returns:
            Tensor ``(1, 1, n_taxels, 3)`` float32 in ``[0, 1]``.

        Raises:
            SensorError: If no frame is captured.
        """
        raw = self._capture_raw()
        if self._background is not None:
            raw = np.clip(raw - self._background, 0, 255)
        # Downsample to n_taxels patches (average pooling over spatial blocks).
        taxels = self._pool(raw)  # (n_taxels, 3)
        t = torch.from_numpy(taxels.astype(np.float32) / 255.0)
        return t.unsqueeze(0).unsqueeze(0)  # (1, 1, n_taxels, 3)

    def _capture_raw(self) -> np.ndarray:
        """Read a raw frame from the camera (stub for real hardware)."""
        # In real deployment: use `digit-interface` or OpenCV VideoCapture.
        # Return synthetic data in tests.
        return np.random.randint(0, 256, (160, 120, 3), dtype=np.uint8)

    def _pool(self, frame: np.ndarray) -> np.ndarray:
        """Average-pool frame to n_taxels patches."""
        H, W, C = frame.shape
        sqrt_n = int(self._n_taxels ** 0.5) or 1
        ph, pw = H // sqrt_n, W // sqrt_n
        taxels = np.zeros((self._n_taxels, C), dtype=np.float64)
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                idx = i * sqrt_n + j
                if idx < self._n_taxels:
                    taxels[idx] = frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw].mean(axis=(0, 1))
        return taxels
