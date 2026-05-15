"""Occlusion wrappers for any BaseEnv.

Wraps an existing environment to programmatically apply varying degrees of
visual occlusion.  Used for the H-A benchmark sweep over 3 occlusion levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from tfwm.sim.base_env import BaseEnv


class OcclusionWrapper(gym.Wrapper):
    """Mask vision observations probabilistically to simulate occlusion.

    Args:
        env: The wrapped :class:`~tfwm.sim.base_env.BaseEnv`.
        occlusion_prob: Fraction of steps where vision is blocked ``∈ [0, 1]``.
        occlusion_mode: ``"blackout"`` zeroes the image; ``"noise"`` replaces
            with uniform noise; ``"drop"`` removes the key from the obs dict.
    """

    def __init__(
        self,
        env: BaseEnv,
        occlusion_prob: float = 0.5,
        occlusion_mode: str = "blackout",
        seed: int = 0,
    ) -> None:
        super().__init__(env)
        self.occlusion_prob = occlusion_prob
        self.occlusion_mode = occlusion_mode
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "vision" in obs and self._rng.random() < self.occlusion_prob:
            if self.occlusion_mode == "blackout":
                obs["vision"] = np.zeros_like(obs["vision"])
            elif self.occlusion_mode == "noise":
                obs["vision"] = self._rng.random(obs["vision"].shape).astype(np.float32)
            elif self.occlusion_mode == "drop":
                del obs["vision"]
            info["camera_queried"] = False
        return obs, reward, terminated, truncated, info


class OcclusionSweep:
    """Helper to evaluate a policy at multiple occlusion levels.

    Args:
        env_factory: Callable returning a fresh :class:`~tfwm.sim.base_env.BaseEnv`.
        levels: Occlusion probabilities to sweep over (default: 3 levels).
    """

    DEFAULT_LEVELS = (0.0, 0.5, 0.9)

    def __init__(
        self,
        env_factory: Any,
        levels: tuple[float, ...] = DEFAULT_LEVELS,
    ) -> None:
        self.env_factory = env_factory
        self.levels = levels

    def envs(self) -> list[tuple[float, OcclusionWrapper]]:
        """Return ``[(level, wrapped_env)]`` for each occlusion level."""
        return [
            (level, OcclusionWrapper(self.env_factory(), occlusion_prob=level))
            for level in self.levels
        ]
