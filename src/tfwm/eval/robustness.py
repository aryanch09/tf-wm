"""Robustness sweep evaluator.

Sweeps over:
  - Occlusion level {0%, 50%, 90%}
  - Tactile noise std {0.0, 0.01, 0.05, 0.1}
  - Tactile sample rate {100, 500, 1000, 5000} Hz
  - Novel object geometries
and reports success rate at each operating point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from tfwm.eval.metrics import success_rate


@dataclass
class RobustnessPoint:
    """One operating point in the robustness sweep."""

    variable: str
    value: float
    success_rate: float
    n_episodes: int


class RobustnessSweep:
    """Evaluate a policy across a sweep of a single environmental variable.

    Args:
        policy_factory: Callable returning a fresh policy.
        env_factory: Callable accepting keyword overrides for the variable
            being swept and returning a gymnasium-compatible environment.
        n_episodes: Episodes per operating point.
        seed: Base seed.
    """

    def __init__(
        self,
        policy_factory: Callable[[], Any],
        env_factory: Callable[..., Any],
        n_episodes: int = 50,
        seed: int = 42,
    ) -> None:
        self.policy_factory = policy_factory
        self.env_factory = env_factory
        self.n_episodes = n_episodes
        self.seed = seed

    def sweep_occlusion(self, levels: tuple[float, ...] = (0.0, 0.5, 0.9)) -> list[RobustnessPoint]:
        """Sweep over visual occlusion probability."""
        return self._sweep("occlusion_prob", levels)

    def sweep_noise(self, stds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1)) -> list[RobustnessPoint]:
        """Sweep over tactile noise standard deviation."""
        return self._sweep("tactile_noise_std", stds)

    def sweep_sample_rate(self, rates: tuple[float, ...] = (100, 500, 1000, 5000)) -> list[RobustnessPoint]:
        """Sweep over tactile acquisition rate."""
        return self._sweep("sample_rate_hz", rates)

    def _sweep(self, variable: str, values: tuple[float, ...]) -> list[RobustnessPoint]:
        points: list[RobustnessPoint] = []
        for val in values:
            env = self.env_factory(**{variable: val})
            policy = self.policy_factory()
            episodes: list[dict[str, Any]] = []
            for ep in range(self.n_episodes):
                obs, _ = env.reset(seed=self.seed + ep)
                policy.reset()
                done = False
                info_last: dict[str, Any] = {"success": False}
                while not done:
                    action = policy.act(obs, deterministic=True)
                    obs, _, terminated, truncated, info_last = env.step(action)
                    done = terminated or truncated
                episodes.append({"success": info_last.get("success", False)})
            sr = success_rate(episodes)
            points.append(RobustnessPoint(variable=variable, value=float(val),
                                          success_rate=sr, n_episodes=self.n_episodes))
        return points

    @staticmethod
    def to_markdown(points: list[RobustnessPoint]) -> str:
        """Format sweep results as Markdown table."""
        header = f"| {points[0].variable} | Success Rate |\n|---|---|\n"
        rows = "".join(f"| {p.value} | {p.success_rate:.3f} |\n" for p in points)
        return header + rows
