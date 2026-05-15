"""Ablation study runner.

Executes the ablation matrix from COPILOT.md Section 9 and produces a
structured results table.  Each ablation swaps out one component of TF-WM
and re-evaluates on the benchmark task suite.

Ablations:
  1. vision-only baseline
  2. symmetric fusion baseline (tactile ≡ vision weight)
  3. tactile-only reactive (no world model)
  4. TF-WM without gating (always query vision)
  5. TF-WM without vision (ever)
  6. TF-WM with tactile=force-magnitude only
  7. TF-WM with varying tactile sample rate {100, 500, 1000, 5000} Hz
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tfwm.eval.metrics import (
    camera_query_rate,
    slip_rate,
    steps_to_success,
    success_rate,
)


@dataclass
class AblationResult:
    """Results for a single ablation variant."""

    name: str
    success_rate_mean: float = 0.0
    success_rate_ci95: float = 0.0
    camera_query_rate_mean: float = 0.0
    slip_rate_mean: float = 0.0
    steps_to_success_mean: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AblationResult({self.name!r}: "
            f"sr={self.success_rate_mean:.3f}±{self.success_rate_ci95:.3f}, "
            f"cqr={self.camera_query_rate_mean:.3f})"
        )


class AblationRunner:
    """Run a suite of ablations and collect results.

    Args:
        ablations: Dict mapping ablation name → ``(model_factory, env_factory)``
            callables.  Each factory returns a ready-to-evaluate object.
        n_episodes: Episodes per ablation variant.
        n_bootstrap: Bootstrap iterations for 95% CI.
        seed: RNG seed.
    """

    def __init__(
        self,
        ablations: dict[str, tuple[Callable[[], Any], Callable[[], Any]]],
        n_episodes: int = 100,
        n_bootstrap: int = 10_000,
        seed: int = 42,
    ) -> None:
        self.ablations = ablations
        self.n_episodes = n_episodes
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(seed)

    def _bootstrap_ci(self, values: list[float], stat: Callable[[np.ndarray], float] = np.mean) -> tuple[float, float]:
        arr = np.array(values)
        boot = np.array([stat(self.rng.choice(arr, len(arr), replace=True)) for _ in range(self.n_bootstrap)])
        return float(boot.mean()), float(np.percentile(boot, 97.5) - np.percentile(boot, 2.5))

    def run_variant(
        self,
        name: str,
        model_factory: Callable[[], Any],
        env_factory: Callable[[], Any],
    ) -> AblationResult:
        """Evaluate one ablation variant.

        Args:
            name: Human-readable variant name.
            model_factory: Returns a policy with ``.act(obs)`` and ``.reset()``.
            env_factory: Returns a gymnasium-compatible env.

        Returns:
            Populated :class:`AblationResult`.
        """
        policy = model_factory()
        env = env_factory()
        episodes: list[dict[str, Any]] = []

        for ep_idx in range(self.n_episodes):
            obs, info = env.reset(seed=ep_idx)
            policy.reset()
            done = False
            ep: dict[str, Any] = {
                "steps": 0, "success": False, "camera_queries": 0,
                "slip_events": 0, "rewards": [],
            }
            while not done:
                action = policy.act(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep["steps"] += 1
                ep["rewards"].append(float(reward))
                ep["camera_queries"] += int(info.get("camera_queried", False))
                ep["slip_events"] += int(info.get("slip_flag", False))
                ep["success"] = bool(info.get("success", False))
            episodes.append(ep)

        sr_list = [float(ep["success"]) for ep in episodes]
        sr_mean, sr_ci = self._bootstrap_ci(sr_list)
        cqr = camera_query_rate(episodes)
        sr_ep = slip_rate(episodes)
        sts = steps_to_success(episodes)

        return AblationResult(
            name=name,
            success_rate_mean=sr_mean,
            success_rate_ci95=sr_ci,
            camera_query_rate_mean=cqr,
            slip_rate_mean=sr_ep,
            steps_to_success_mean=sts,
        )

    def run_all(self) -> dict[str, AblationResult]:
        """Run all registered ablation variants.

        Returns:
            Dict mapping variant name → :class:`AblationResult`.
        """
        results: dict[str, AblationResult] = {}
        for name, (model_f, env_f) in self.ablations.items():
            results[name] = self.run_variant(name, model_f, env_f)
        return results

    @staticmethod
    def to_markdown(results: dict[str, AblationResult]) -> str:
        """Render results as a Markdown table."""
        header = "| Variant | SR (mean) | SR CI95 | CQR | SLR | STS |\n"
        sep    = "|---------|-----------|---------|-----|-----|-----|\n"
        rows = "".join(
            f"| {r.name} | {r.success_rate_mean:.3f} | ±{r.success_rate_ci95:.3f} "
            f"| {r.camera_query_rate_mean:.3f} | {r.slip_rate_mean:.3f} "
            f"| {r.steps_to_success_mean:.1f} |\n"
            for r in results.values()
        )
        return header + sep + rows
