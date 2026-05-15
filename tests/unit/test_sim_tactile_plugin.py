"""Tests for tactile simulator plugin utilities."""

from __future__ import annotations

import torch
from tfwm.sim.tactile_plugin import TactileNoiseConfig, apply_tactile_noise


def test_apply_tactile_noise_shape_and_bounds() -> None:
    tactile = torch.ones(2, 3, 4, 1) * 0.5
    noisy, drift = apply_tactile_noise(
        tactile,
        TactileNoiseConfig(
            gaussian_std=0.0,
            dropout_p=0.0,
            drift_std=0.0,
            saturation=1.0,
            hysteresis=0.0,
        ),
    )
    assert noisy.shape == tactile.shape
    assert drift.shape == tactile.shape
    assert torch.all((noisy >= 0.0) & (noisy <= 1.0))
