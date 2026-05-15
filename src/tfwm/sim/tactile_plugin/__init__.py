"""Tactile contact sensor plugins for simulators."""

from __future__ import annotations

from tfwm.sim.tactile_plugin.mjx_taxel import TactileNoiseConfig, apply_tactile_noise

__all__ = ["TactileNoiseConfig", "apply_tactile_noise"]
