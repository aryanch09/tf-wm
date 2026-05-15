"""Gymnasium-compatible base environment contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class BaseEnv(gym.Env, ABC):
    """Base environment contract for tactile-first manipulation tasks."""

    metadata = {"render_modes": []}

    @abstractmethod
    def requires_vision(self, t: int) -> bool:
        """Return whether the observation at timestep ``t`` should include vision."""

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the environment."""

    @abstractmethod
    def step(self, action):
        """Advance the environment one step."""
