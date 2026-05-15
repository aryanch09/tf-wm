"""RGB-D camera abstractions."""

from __future__ import annotations

import numpy as np
import torch


class RGBCamera:
    """RGB camera abstraction.

    Args:
        device_id: Camera index.
        resolution: ``(H, W)`` output resolution.
        sample_rate_hz: Target frame rate.
    """

    def __init__(self, device_id: int = 0, resolution: tuple[int, int] = (224, 224), sample_rate_hz: int = 30) -> None:
        self.device_id = device_id
        self.resolution = resolution
        self.sample_rate_hz = sample_rate_hz

    def read(self) -> torch.Tensor:
        """Return one RGB frame ``(1, 3, H, W)`` float32 in ``[0, 1]``."""
        H, W = self.resolution
        arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        return t.unsqueeze(0)


class RGBDCamera(RGBCamera):
    """RGB-D camera (adds depth channel).

    Args:
        max_depth_m: Maximum depth range in metres (for normalisation).
    """

    def __init__(self, max_depth_m: float = 2.0, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.max_depth_m = max_depth_m

    def read_rgbd(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rgb [1,3,H,W], depth [1,1,H,W])`` float32."""
        rgb = super().read()
        H, W = self.resolution
        depth = torch.rand(1, 1, H, W)
        return rgb, depth
