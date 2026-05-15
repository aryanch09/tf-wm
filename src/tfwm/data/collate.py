"""Collation functions for variable-length tactile sequences."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_episodes(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate batch of episode tensors.

    Args:
        batch: List of tensor dicts from episodes.

    Returns:
        Collated batch dict.
    """
    # For now, assume all episodes have same length (padded)
    # In future, handle variable lengths
    collated = {}
    for key in batch[0].keys():
        tensors = [item[key] for item in batch]
        if key in ["tactile", "vision", "proprio", "actions"]:
            # Stack along batch dimension
            collated[key] = torch.stack(tensors)
        else:
            # Stack scalars/rewards
            collated[key] = torch.stack(tensors)

    return collated


def collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate batch of sequence tensors.

    Args:
        batch: List of tensor dicts from sequences.

    Returns:
        Collated batch dict.
    """
    # Sequences should already be same length
    return collate_episodes(batch)


def collate_variable_length(
    batch: list[dict[str, torch.Tensor]],
    pad_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Collate batch with variable-length sequences.

    Args:
        batch: List of tensor dicts.
        pad_value: Value to pad with.

    Returns:
        Collated batch with masks.
    """
    collated = {}

    # Get all keys
    keys = batch[0].keys()

    for key in keys:
        tensors = [item[key] for item in batch]
        if tensors[0].dim() >= 2:  # Sequence tensors
            collated[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            collated[f"{key}_mask"] = torch.tensor(
                [[1] * len(t) + [0] * (collated[key].shape[1] - len(t)) for t in tensors]
            )
        else:  # Scalar tensors
            collated[key] = torch.stack(tensors)

    return collated
