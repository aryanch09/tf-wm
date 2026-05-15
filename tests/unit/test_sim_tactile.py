"""Tests for tactile processing utilities."""

from __future__ import annotations

import torch
from tfwm.sim.tactile import (
    normalize_tactile,
    tactile_centroid,
    tactile_contact_mask,
    tactile_curvature,
    tactile_gradient,
)


class TestTactileProcessing:
    """Test tactile processing functions."""

    def test_normalize_tactile(self):
        """Test tactile normalization."""
        tactile = torch.tensor([[0.0, 0.5], [1.0, 1.5]])
        normalized = normalize_tactile(tactile, min_val=0.0, max_val=1.0)

        assert normalized.shape == tactile.shape
        assert torch.all(normalized >= 0.0)
        assert torch.all(normalized <= 1.0)
        assert normalized[0, 0].item() == 0.0
        assert normalized[0, 1].item() == 0.5
        assert normalized[1, 0].item() == 1.0
        assert normalized[1, 1].item() == 1.0  # clipped

    def test_tactile_gradient(self):
        """Test gradient computation."""
        tactile = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        gradient = tactile_gradient(tactile)

        assert gradient.shape == tactile.shape
        assert gradient[1, 1].item() == 0.0
        assert gradient[1, 0].item() > 0.0
        assert gradient[1, 2].item() > 0.0
        assert gradient[0, 1].item() > 0.0
        assert gradient[2, 1].item() > 0.0

    def test_tactile_curvature(self):
        """Test curvature computation."""
        tactile = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        curvature = tactile_curvature(tactile)

        assert curvature.shape == tactile.shape
        assert curvature[1, 1].item() < 0.0

    def test_tactile_contact_mask(self):
        """Test contact mask generation."""
        tactile = torch.tensor([[0.05, 0.15], [0.25, 0.05]])
        mask = tactile_contact_mask(tactile, threshold=0.1)

        assert mask.shape == tactile.shape
        assert mask[0, 0].item() == 0.0
        assert mask[0, 1].item() == 1.0
        assert mask[1, 0].item() == 1.0
        assert mask[1, 1].item() == 0.0

    def test_tactile_centroid(self):
        """Test centroid computation."""
        # Uniform contact
        tactile = torch.ones((4, 4))
        cy, cx = tactile_centroid(tactile)
        assert cy == 1.5
        assert cx == 1.5

        # No contact
        tactile = torch.zeros((4, 4))
        cy, cx = tactile_centroid(tactile)
        assert cy == 2.0
        assert cx == 2.0

        # Corner contact
        tactile = torch.zeros((4, 4))
        tactile[0, 0] = 1.0
        cy, cx = tactile_centroid(tactile)
        assert cy == 0.0
        assert cx == 0.0
