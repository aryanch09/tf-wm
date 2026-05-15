"""Scheduled sampling decay for teacher-forcing probability."""

from __future__ import annotations

import math


class ScheduledSamplingCallback:
    """Decay the teacher-forcing probability ``ε`` over training.

    Supports linear, exponential, and inverse-sigmoid schedules.

    Args:
        start: Initial teacher-forcing rate (1.0 = fully teacher-forced).
        end: Final teacher-forcing rate.
        decay_steps: Number of steps over which to decay.
        schedule: One of ``"linear"``, ``"exponential"``, ``"sigmoid"``.

    Usage::

        ss = ScheduledSamplingCallback(start=1.0, end=0.1, decay_steps=100_000)
        for step in range(max_steps):
            epsilon = ss.get(step)
            # use teacher forcing with prob epsilon, free running with 1-epsilon
    """

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.1,
        decay_steps: int = 100_000,
        schedule: str = "linear",
    ) -> None:
        self.start = start
        self.end = end
        self.decay_steps = max(decay_steps, 1)
        self.schedule = schedule
        _valid = {"linear", "exponential", "sigmoid"}
        if schedule not in _valid:
            raise ValueError(f"schedule must be one of {_valid}, got '{schedule}'")

    def get(self, step: int) -> float:
        """Return the teacher-forcing probability at ``step``.

        Args:
            step: Current training step.

        Returns:
            ``ε ∈ [end, start]``.
        """
        frac = min(step / self.decay_steps, 1.0)
        if self.schedule == "linear":
            return self.start - frac * (self.start - self.end)
        if self.schedule == "exponential":
            k = math.log(max(self.end / max(self.start, 1e-8), 1e-8))
            return self.start * math.exp(k * frac)
        # sigmoid: ε = k / (k + exp(step / k))  where k = decay_steps // 4
        k = max(self.decay_steps // 4, 1)
        return k / (k + math.exp(step / k))
