"""Shape utilities and einops helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def assert_shape(tensor: Tensor, expected_shape: str) -> None:
    """Assert tensor shape matches expected pattern."""
    expected = expected_shape.split()
    if tensor.dim() != len(expected):
        raise ValueError(
            f"Expected tensor of rank {len(expected)} for shape '{expected_shape}', got {tensor.dim()}"
        )


def pack_sequences(sequences: list[Tensor], pad_value: float = 0.0) -> tuple[Tensor, Tensor]:
    """Pack variable-length sequences into padded tensor.

    Args:
        sequences: List of tensors of shape (T_i, ...).
        pad_value: Value to pad with.

    Returns:
        Tuple of (padded_tensor [B, T_max, ...], lengths [B]).
    """
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = lengths.max().item()
    batch_size = len(sequences)

    # Assume all sequences have same shape except first dim
    sample_shape = sequences[0].shape[1:]
    padded = torch.full((batch_size, max_len, *sample_shape), pad_value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded[i, : lengths[i]] = seq

    return padded, lengths


def unpack_sequences(padded: Tensor, lengths: Tensor) -> list[Tensor]:
    """Unpack padded tensor into variable-length sequences.

    Args:
        padded: Padded tensor of shape (B, T_max, ...).
        lengths: Lengths tensor of shape (B,).

    Returns:
        List of tensors of shape (T_i, ...).
    """
    sequences = []
    for i, length in enumerate(lengths):
        sequences.append(padded[i, :length])
    return sequences


def temporal_crop(tensor: Tensor, start: int, length: int) -> Tensor:
    """Crop tensor temporally.

    Args:
        tensor: Input tensor of shape (..., T, ...).
        start: Start index.
        length: Crop length.

    Returns:
        Cropped tensor.
    """
    return tensor[..., start : start + length, :]


def batch_repeat(tensor: Tensor, repeats: int) -> Tensor:
    """Repeat tensor along batch dimension.

    Args:
        tensor: Input tensor of shape (B, ...).
        repeats: Number of repeats.

    Returns:
        Repeated tensor of shape (B * repeats, ...).
    """
    return torch.cat([tensor] * repeats, dim=0)
