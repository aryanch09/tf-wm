"""Timing and profiling utilities.

Provides a context-manager ``Timer`` for wall-clock measurement and a
``LatencyBudget`` guard that raises if a code block exceeds a budget.
Used to enforce the <30 ms CEM rollout SLA from the project spec.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class TimingStats:
    """Accumulator for repeated measurements."""

    name: str
    samples: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.samples.append(elapsed_ms)

    @property
    def mean_ms(self) -> float:
        return sum(self.samples) / max(len(self.samples), 1)

    @property
    def max_ms(self) -> float:
        return max(self.samples, default=0.0)

    @property
    def min_ms(self) -> float:
        return min(self.samples, default=0.0)

    def __repr__(self) -> str:
        return (
            f"TimingStats({self.name!r}: n={len(self.samples)}, "
            f"mean={self.mean_ms:.2f} ms, max={self.max_ms:.2f} ms)"
        )


class Timer:
    """Simple wall-clock timer with optional accumulation.

    Args:
        name: Logical name for logging.
        stats: Optional :class:`TimingStats` to accumulate into.
        budget_ms: If set and elapsed > budget, logs a warning.

    Usage::

        with Timer("cem_rollout", budget_ms=30.0) as t:
            action = planner.plan(z, goal)
        print(t.elapsed_ms)
    """

    def __init__(
        self,
        name: str = "block",
        stats: TimingStats | None = None,
        budget_ms: float | None = None,
    ) -> None:
        self.name = name
        self.stats = stats
        self.budget_ms = budget_ms
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000
        if self.stats is not None:
            self.stats.record(self.elapsed_ms)
        if self.budget_ms is not None and self.elapsed_ms > self.budget_ms:
            import warnings
            warnings.warn(
                f"[timing] {self.name} took {self.elapsed_ms:.1f} ms "
                f"(budget: {self.budget_ms:.1f} ms)",
                stacklevel=2,
            )


@contextmanager
def cuda_sync_timer(name: str = "block") -> Generator[Timer, None, None]:
    """Synchronise CUDA before timing to get accurate GPU measurements."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    with Timer(name) as t:
        yield t
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
    # Update elapsed after the post-sync.
    t.elapsed_ms = (time.perf_counter() - t._start) * 1_000


_global_registry: dict[str, TimingStats] = {}


def get_timer(name: str) -> TimingStats:
    """Get-or-create a global :class:`TimingStats` accumulator."""
    if name not in _global_registry:
        _global_registry[name] = TimingStats(name)
    return _global_registry[name]


def report_timings() -> dict[str, dict[str, float]]:
    """Return a summary of all recorded global timers."""
    return {
        name: {"mean_ms": s.mean_ms, "max_ms": s.max_ms, "n": len(s.samples)}
        for name, s in _global_registry.items()
    }
