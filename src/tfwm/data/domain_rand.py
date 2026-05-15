"""Domain randomization utilities."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class DomainRandomizer:
    """Applies domain randomization to simulation parameters."""

    def __init__(
        self,
        friction_range: tuple[float, float] = (0.3, 1.2),
        mass_scale_range: tuple[float, float] = (0.7, 1.3),
        object_scale_range: tuple[float, float] = (0.9, 1.1),
        lighting_range: tuple[float, float] = (0.5, 1.5),
    ):
        self.friction_range = friction_range
        self.mass_scale_range = mass_scale_range
        self.object_scale_range = object_scale_range
        self.lighting_range = lighting_range

    def sample(self) -> dict[str, float]:
        """Sample randomization parameters.

        Returns:
            Dict of randomized parameters.
        """
        return {
            "friction": random.uniform(*self.friction_range),
            "mass_scale": random.uniform(*self.mass_scale_range),
            "object_scale": random.uniform(*self.object_scale_range),
            "lighting": random.uniform(*self.lighting_range),
        }

    def apply_to_env(self, env: Any, params: dict[str, float]) -> None:
        """Apply randomization to environment.

        Args:
            env: Environment instance.
            params: Randomization parameters.
        """
        if hasattr(env, "set_domain_parameters"):
            env.set_domain_parameters(params)
            return
        if hasattr(env, "domain_parameters"):
            env.domain_parameters.update(params)
            return
        for name, value in params.items():
            if hasattr(env, name):
                setattr(env, name, value)


def randomize_tactile_calibration(
    base_calibration: np.ndarray,
    offset_range: tuple[float, float] = (-0.1, 0.1),
) -> np.ndarray:
    """Randomize tactile sensor calibration.

    Args:
        base_calibration: Base calibration values.
        offset_range: Range for additive offsets.

    Returns:
        Randomized calibration.
    """
    offsets = np.random.uniform(*offset_range, size=base_calibration.shape)
    return base_calibration + offsets
