"""Replay buffer with prioritized sampling."""

from __future__ import annotations

import random

import numpy as np
import torch

from .episode import Episode


class ReplayBuffer:
    """Replay buffer for episodes."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.episodes: list[Episode] = []
        self.priorities: list[float] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: Episode, priority: float = 1.0) -> None:
        """Add episode to buffer.

        Args:
            episode: Episode to add.
            priority: Sampling priority.
        """
        if len(self) >= self.capacity:
            # Remove oldest
            self.episodes.pop(0)
            self.priorities.pop(0)

        self.episodes.append(episode)
        self.priorities.append(priority)

    def sample_batch(self, batch_size: int) -> list[Episode]:
        """Sample batch of episodes.

        Args:
            batch_size: Number of episodes to sample.

        Returns:
            List of sampled episodes.
        """
        if len(self) == 0:
            raise ValueError("Buffer is empty")

        # Simple uniform sampling for now
        indices = random.choices(range(len(self)), k=batch_size)
        return [self.episodes[i] for i in indices]

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities for episodes.

        Args:
            indices: Episode indices.
            priorities: New priorities.
        """
        for idx, prio in zip(indices, priorities, strict=False):
            self.priorities[idx] = prio

    def get_dataset(self) -> EpisodeDataset:
        """Get torch Dataset wrapper.

        Returns:
            Dataset for torch DataLoader.
        """
        return EpisodeDataset(self.episodes)


class EpisodeDataset(torch.utils.data.Dataset):
    """Torch Dataset for episodes."""

    def __init__(self, episodes: list[Episode]):
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get episode as tensor dict.

        Args:
            idx: Episode index.

        Returns:
            Tensor dict.
        """
        return self.episodes[idx].to_tensors()


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        super().__init__(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def add_episode(self, episode: Episode, priority: float | None = None) -> None:
        """Add episode with priority.

        Args:
            episode: Episode to add.
            priority: Priority (uses max_priority if None).
        """
        if priority is None:
            priority = self.max_priority
        super().add_episode(episode, priority)
        self.max_priority = max(self.max_priority, priority)

    def sample_batch(self, batch_size: int) -> tuple[list[Episode], list[int], list[float]]:
        """Sample batch with priorities.

        Args:
            batch_size: Batch size.

        Returns:
            Tuple of (episodes, indices, weights).
        """
        if len(self) == 0:
            raise ValueError("Buffer is empty")

        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self), size=batch_size, p=probs)
        episodes = [self.episodes[i] for i in indices]

        # Importance sampling weights
        weights = (len(self) * probs[indices]) ** (-self.alpha)
        weights /= weights.max()

        return episodes, indices.tolist(), weights.tolist()

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities and max_priority.

        Args:
            indices: Episode indices.
            priorities: New priorities.
        """
        super().update_priorities(indices, priorities)
        self.max_priority = max(self.max_priority, max(priorities))
