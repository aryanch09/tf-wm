"""Episode dataclass and I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import TypeAdapter

from .schema import Episode as EpisodeSchema


def _pad_2d_arrays(arrays: list[np.ndarray]) -> torch.Tensor:
    """Pad a list of ``(N, C)`` arrays into a dense tensor."""

    max_rows = max(array.shape[0] for array in arrays)
    max_cols = max(array.shape[1] for array in arrays)
    tensor = torch.zeros(len(arrays), max_rows, max_cols, dtype=torch.float32)
    for index, array in enumerate(arrays):
        rows, cols = array.shape
        tensor[index, :rows, :cols] = torch.from_numpy(array.astype(np.float32, copy=False))
    return tensor


def _pad_1d_arrays(arrays: list[np.ndarray]) -> torch.Tensor:
    """Pad variable-width vectors into a dense tensor."""

    width = max(array.size for array in arrays)
    tensor = torch.zeros(len(arrays), width, dtype=torch.float32)
    for index, array in enumerate(arrays):
        tensor[index, : array.size] = torch.from_numpy(array.astype(np.float32, copy=False))
    return tensor


def _vision_tensor(values: list[float] | None) -> torch.Tensor:
    """Convert optional flattened vision data to ``(3, 64, 64)``."""

    tensor = torch.zeros(3, 64, 64, dtype=torch.float32)
    if not values:
        return tensor
    flat = torch.tensor(values, dtype=torch.float32).flatten()
    n = min(flat.numel(), tensor.numel())
    tensor.flatten()[:n] = flat[:n]
    return tensor


class Episode:
    """Episode data with tensor conversions."""

    def __init__(self, schema: EpisodeSchema):
        self.schema = schema
        self._tensors: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.schema.n_steps

    @property
    def metadata(self) -> Any:
        return self.schema.metadata

    def to_tensors(self, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
        """Convert episode to tensor dict.

        Args:
            device: Device to put tensors on.

        Returns:
            Dict with 'tactile', 'vision', 'proprio', 'actions', etc.
        """
        if self._tensors is not None:
            return {k: v.to(device) for k, v in self._tensors.items()}

        # Tactile: (T, N_taxel, C)
        tactile_data = []
        for obs in self.schema.observations:
            if obs.tactile:
                tactile_data.append(np.array(obs.tactile, dtype=np.float32).reshape(-1, 1))
            else:
                tactile_data.append(np.zeros((1, 1), dtype=np.float32))
        tactile = _pad_2d_arrays(tactile_data)

        # Vision: (T, C, H, W)
        vision = torch.stack([_vision_tensor(obs.vision) for obs in self.schema.observations])

        # Proprio: (T, d_p)
        proprio_data = [
            np.asarray(obs.proprio, dtype=np.float32)
            if obs.proprio
            else np.zeros(6, dtype=np.float32)
            for obs in self.schema.observations
        ]
        proprio = _pad_1d_arrays(proprio_data)

        # Actions: (T, d_a)
        actions = _pad_1d_arrays(
            [np.asarray(action.action, dtype=np.float32) for action in self.schema.actions]
        )

        # Rewards: (T,)
        rewards = torch.tensor(self.schema.rewards, dtype=torch.float32)

        # Dones: (T,)
        dones = torch.tensor(self.schema.dones, dtype=torch.bool)

        self._tensors = {
            "tactile": tactile,
            "vision": vision,
            "proprio": proprio,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

        return {k: v.to(device) for k, v in self._tensors.items()}

    @classmethod
    def from_json(cls, path: Path) -> Episode:
        """Load episode from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Episode instance.
        """
        with open(path) as f:
            data = json.load(f)
        schema = TypeAdapter(EpisodeSchema).validate_python(data)
        return cls(schema)

    def to_json(self, path: Path) -> None:
        """Save episode to JSON file.

        Args:
            path: Path to save to.
        """
        with open(path, "w") as f:
            json.dump(self.schema.model_dump(), f, indent=2)

    @classmethod
    def from_parquet(cls, path: Path) -> Episode:
        """Load episode from Parquet file.

        Args:
            path: Path to Parquet file.

        Returns:
            Episode instance.
        """
        from .loaders.parquet import load_episodes_from_parquet

        episodes = load_episodes_from_parquet(path)
        if len(episodes) != 1:
            raise ValueError(f"Expected exactly one episode in {path}, found {len(episodes)}")
        return episodes[0]

    def to_parquet(self, path: Path) -> None:
        """Save episode to Parquet file.

        Args:
            path: Path to save to.
        """
        from .loaders.parquet import save_episodes_to_parquet

        save_episodes_to_parquet([self], path)
