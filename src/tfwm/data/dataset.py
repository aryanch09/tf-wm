"""Torch Dataset wrappers for episodes."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .episode import Episode


class EpisodeDataset(Dataset):
    """Dataset of episodes."""

    def __init__(self, episodes: list[Episode]):
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.episodes[idx].to_tensors()


class DirectoryDataset(Dataset):
    """Dataset from directory of episode files."""

    def __init__(self, data_dir: Path, file_pattern: str = "*.json"):
        self.files = list(data_dir.glob(file_pattern))
        self.files.sort()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode = Episode.from_json(self.files[idx])
        return episode.to_tensors()


class SequenceDataset(Dataset):
    """Dataset that yields fixed-length sequences from episodes."""

    def __init__(
        self,
        episodes: list[Episode],
        seq_length: int,
        stride: int = 1,
        min_episode_length: int = 10,
    ):
        self.sequences = []
        for episode in episodes:
            if len(episode) < min_episode_length:
                continue
            for start in range(0, len(episode) - seq_length + 1, stride):
                self.sequences.append((episode, start, start + seq_length))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode, start, end = self.sequences[idx]
        tensors = episode.to_tensors()

        # Slice to sequence
        result = {}
        for key, tensor in tensors.items():
            if tensor.dim() >= 2:  # Has time dimension
                result[key] = tensor[start:end]
            else:
                result[key] = tensor

        return result
