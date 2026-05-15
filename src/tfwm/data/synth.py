"""Synthetic tactile data generators and torch Dataset.

The :class:`SyntheticTactileDataset` class is the primary entry-point used by
the Phase 1 trainer when no real data is available.  It produces plausible
tactile streams with smooth contact dynamics, random actions, and optional
contact-event labels.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..types import Tactile


def generate_synthetic_tactile_episode(
    length: int = 100,
    n_taxels: int = 16,
    n_channels: int = 1,
    contact_pattern: str = "random",
) -> Tactile:
    """Generate synthetic tactile episode.

    Args:
        length: Episode length in timesteps.
        n_taxels: Number of taxels.
        n_channels: Number of channels per taxel.
        contact_pattern: Pattern type ("random", "smooth", "step").

    Returns:
        Synthetic tactile tensor (1, length, n_taxels, n_channels).
    """
    if contact_pattern == "random":
        tactile = torch.randn(1, length, n_taxels, n_channels) * 0.1
    elif contact_pattern == "smooth":
        # Smooth contact evolution
        base = torch.randn(1, 1, n_taxels, n_channels)
        noise = torch.randn(1, length, n_taxels, n_channels) * 0.05
        tactile = base + torch.cumsum(noise, dim=1) * 0.1
    elif contact_pattern == "step":
        # Step function contact
        tactile = torch.zeros(1, length, n_taxels, n_channels)
        step_time = length // 2
        tactile[:, step_time:, :, :] = torch.randn(1, 1, n_taxels, n_channels) * 0.2
    else:
        raise ValueError(f"Unknown pattern: {contact_pattern}")

    # Ensure non-negative
    tactile = torch.clamp(tactile, 0, 1)

    return tactile


def generate_contact_events(tactile: Tactile, threshold: float = 0.1) -> Tensor:
    """Generate contact events from tactile data.

    Args:
        tactile: Tactile tensor.
        threshold: Contact threshold.

    Returns:
        Contact events tensor (batch, time).
    """
    # Simple threshold-based contact detection
    max_force = tactile.max(dim=-1)[0].max(dim=-1)[0]  # (batch, time)
    return (max_force > threshold).long()


def generate_slip_events(tactile: Tactile, window: int = 5) -> Tensor:
    """Generate slip events from tactile changes.

    Args:
        tactile: Tactile tensor.
        window: Window size for change detection.

    Returns:
        Slip events tensor (batch, time).
    """
    # Detect rapid changes in contact
    diff = torch.diff(tactile, dim=1)  # (batch, time-1, n_taxels, channels)
    max_diff = diff.abs().max(dim=-1)[0].max(dim=-1)[0]  # (batch, time-1)

    # Pad to original length
    slip = torch.zeros_like(tactile[:, :, 0, 0])
    slip[:, 1:] = (max_diff > 0.05).float()  # Threshold for slip

    return slip


class SyntheticTactileDataset(Dataset):
    """In-memory synthetic tactile dataset for smoke tests and unit tests.

    Each item is a dict with keys::

        tactile   (T, N, C)   — float32
        actions   (T, d_a)    — float32
        contact_events (T,)   — int64 category indices
        proprio   (T, d_p)    — float32

    Args:
        n_episodes: Number of episodes to generate.
        episode_length: Number of timesteps per episode.
        n_taxels: Taxels per timestep.
        n_channels: Channels per taxel.
        d_action: Action dimensionality.
        d_proprio: Proprioception dimensionality.
        n_contact_classes: Number of contact-event categories.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_episodes: int = 100,
        episode_length: int = 50,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_action: int = 6,
        d_proprio: int = 12,
        n_contact_classes: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        T, N, C, A, P = episode_length, n_taxels, n_channels, d_action, d_proprio

        self._data: list[dict[str, torch.Tensor]] = []
        for _ in range(n_episodes):
            # Smooth contact via cumulative noise.
            noise = torch.randn(T, N, C, generator=rng) * 0.05
            tactile = torch.cumsum(noise, dim=0).clamp(0, 1)
            actions = torch.randn(T, A, generator=rng) * 0.3
            proprio = torch.randn(T, P, generator=rng) * 0.1
            max_force = tactile.max(dim=-1)[0].max(dim=-1)[0]  # (T,)
            contact_events = torch.zeros(T, dtype=torch.long)
            contact_events[max_force > 0.3] = 1   # stable
            contact_events[max_force > 0.6] = 2   # slip
            self._data.append({
                "tactile": tactile,
                "actions": actions,
                "proprio": proprio,
                "contact_events": contact_events,
            })

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._data[idx]
