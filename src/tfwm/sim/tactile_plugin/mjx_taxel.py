"""MJX-compatible tactile taxel noise model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TactileNoiseConfig:
    """Noise and nonlinearity settings for simulated tactile streams."""

    gaussian_std: float = 0.01
    dropout_p: float = 0.05
    drift_alpha: float = 0.98
    drift_std: float = 0.001
    saturation: float = 1.0
    hysteresis: float = 0.02
    jitter_ms: float = 2.0


def apply_tactile_noise(
    tactile: torch.Tensor,
    config: TactileNoiseConfig | None = None,
    drift_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Section 10 tactile sim-to-real noise model.

    Args:
        tactile: Tactile tensor of any shape.
        config: Noise configuration.
        drift_state: Optional previous Markov drift state.

    Returns:
        Tuple of noisy tactile tensor and updated drift state.
    """

    cfg = config or TactileNoiseConfig()
    if drift_state is None:
        drift_state = torch.zeros_like(tactile)
    drift = cfg.drift_alpha * drift_state + cfg.drift_std * torch.randn_like(tactile)
    noisy = tactile + drift + cfg.gaussian_std * torch.randn_like(tactile)
    dropout = torch.rand_like(tactile) < cfg.dropout_p
    noisy = noisy.masked_fill(dropout, 0.0)
    if cfg.hysteresis > 0.0:
        noisy = torch.where(noisy.abs() < cfg.hysteresis, torch.zeros_like(noisy), noisy)
    noisy = noisy.clamp(0.0, cfg.saturation)
    return noisy, drift
