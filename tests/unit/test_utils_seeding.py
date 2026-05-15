"""Tests for seeding utilities."""

import torch
from tfwm.utils.seeding import get_rng_state, set_rng_state, set_seed


def test_set_seed_reproducibility():
    """Test that setting seed produces reproducible results."""
    set_seed(42)
    a = torch.randn(10)

    set_seed(42)
    b = torch.randn(10)

    assert torch.allclose(a, b)


def test_rng_state_save_restore():
    """Test saving and restoring RNG state."""
    set_seed(42)
    state = get_rng_state()

    val1 = torch.randn(5)

    # Change state
    torch.randn(10)

    # Restore
    set_rng_state(state)
    val2 = torch.randn(5)

    assert torch.allclose(val1, val2)


def test_worker_init_fn():
    """Test worker initialization function."""
    from tfwm.utils.seeding import worker_init_fn

    # Should not raise
    worker_init_fn(0)
    worker_init_fn(1)
