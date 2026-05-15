"""Core type definitions for TF-WM.

The aliases in this module intentionally stay importable without optional
runtime shape-checking dependencies. When :mod:`jaxtyping` is installed they
carry the shape strings from the project contract; otherwise they degrade to
plain ``torch.Tensor`` aliases so lightweight tooling can still import TF-WM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import torch

try:  # pragma: no cover - exercised only when jaxtyping is installed.
    from jaxtyping import Float, Int

    Tactile = Float[torch.Tensor, "batch time taxel channel"]
    TactileLatent = Float[torch.Tensor, "batch time d_z"]
    Vision = Float[torch.Tensor, "batch time c h w"]
    Proprio = Float[torch.Tensor, "batch time d_p"]
    Action = Float[torch.Tensor, "batch time d_a"]
    ContactEvent = Int[torch.Tensor, "batch time"]
    GateScore = Float[torch.Tensor, "batch time"]
    AffordanceLogits = Float[torch.Tensor, "batch n_affordance"]
except Exception:  # pragma: no cover - keeps local imports dependency-light.
    Tactile = torch.Tensor
    TactileLatent = torch.Tensor
    Vision = torch.Tensor
    Proprio = torch.Tensor
    Action = torch.Tensor
    ContactEvent = torch.Tensor
    GateScore = torch.Tensor
    AffordanceLogits = torch.Tensor


class LatentDist(TypedDict):
    """Distribution over tactile latents."""

    mu: torch.Tensor
    logvar: torch.Tensor


@dataclass(frozen=True, slots=True)
class WorldModelOutput:
    """Output of the world model forward pass."""

    latent: LatentDist
    pred_tactile: Tactile | None
    pred_contact: ContactEvent | None
    pred_affordance: AffordanceLogits | None
    pred_object_state: torch.Tensor | None
    gate_score: GateScore


@runtime_checkable
class TactileSensor(Protocol):
    """Protocol for tactile sensors."""

    sample_rate_hz: int
    n_taxels: int
    n_channels: int

    def read(self) -> Tactile:
        """Read current tactile measurement.

        Returns:
            Tactile tensor of shape (1, 1, n_taxels, n_channels).
        """
        ...

    def calibrate(self) -> None:
        """Calibrate the sensor."""
        ...


@runtime_checkable
class Dynamics(Protocol):
    """Protocol for dynamics models."""

    def rollout(
        self,
        z0: TactileLatent,
        actions: Action,
        context: torch.Tensor | None = None,
    ) -> tuple[TactileLatent, dict[str, torch.Tensor]]:
        """Roll out dynamics from initial latent.

        Args:
            z0: Initial latent state, shape (B, T0, d_z).
            actions: Action sequence, shape (B, H, d_a).
            context: Optional context tensor.

        Returns:
            Tuple of (predicted latents [B, H, d_z], auxiliary outputs dict).
        """
        ...
