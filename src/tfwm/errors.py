"""Project-specific exceptions."""

from __future__ import annotations


class TFWMError(Exception):
    """Base class for TF-WM errors."""


class ShapeError(TFWMError, ValueError):
    """Raised when a tensor has an invalid shape for a contract."""


class ConfigError(TFWMError, ValueError):
    """Raised when a runtime or Hydra configuration is invalid."""


class SensorError(TFWMError, RuntimeError):
    """Raised when a tactile or vision sensor cannot produce valid data."""


class SafetyError(TFWMError, RuntimeError):
    """Raised when a control command violates a safety constraint."""
