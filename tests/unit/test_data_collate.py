"""Tests for collation functions."""

import torch
from tfwm.data.collate import collate_episodes, collate_sequences, collate_variable_length


def test_collate_episodes():
    """Test collating episode tensors."""
    batch = [
        {
            "tactile": torch.randn(5, 10, 3),
            "actions": torch.randn(5, 2),
            "rewards": torch.randn(5),
        },
        {
            "tactile": torch.randn(5, 10, 3),
            "actions": torch.randn(5, 2),
            "rewards": torch.randn(5),
        },
    ]

    collated = collate_episodes(batch)

    assert collated["tactile"].shape == (2, 5, 10, 3)
    assert collated["actions"].shape == (2, 5, 2)
    assert collated["rewards"].shape == (2, 5)


def test_collate_sequences():
    """Test collating sequence tensors."""
    batch = [
        {
            "tactile": torch.randn(10, 10, 3),
            "actions": torch.randn(10, 2),
        },
        {
            "tactile": torch.randn(10, 10, 3),
            "actions": torch.randn(10, 2),
        },
    ]

    collated = collate_sequences(batch)

    assert collated["tactile"].shape == (2, 10, 10, 3)
    assert collated["actions"].shape == (2, 10, 2)


def test_collate_variable_length():
    """Test collating variable-length sequences."""
    batch = [
        {
            "tactile": torch.randn(5, 10, 3),
            "actions": torch.randn(5, 2),
        },
        {
            "tactile": torch.randn(3, 10, 3),
            "actions": torch.randn(3, 2),
        },
    ]

    collated = collate_variable_length(batch)

    assert collated["tactile"].shape == (2, 5, 10, 3)  # Padded to max length
    assert "tactile_mask" in collated
    assert collated["tactile_mask"].shape == (2, 5)
    assert collated["actions"].shape == (2, 5, 2)  # Also padded
