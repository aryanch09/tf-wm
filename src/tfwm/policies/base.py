"""Base policy protocol and abstract class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Policy(Protocol):
    """Protocol for all TF-WM policies."""

    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return an action given the current observation dict.

        Args:
            obs: Dict containing at minimum ``"tactile"`` key.
            deterministic: If ``True``, return the mode (no sampling).

        Returns:
            Action tensor ``(B, d_a)`` or ``(d_a,)`` for single-env.
        """
        ...

    def reset(self) -> None:
        """Reset any recurrent state at the start of an episode."""
        ...


class PolicyBase(nn.Module):
    """Abstract base class for trainable policies."""

    @abstractmethod
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Return action for the given observation."""

    def reset(self) -> None:
        """Reset recurrent state."""
