"""Tactile processing utilities."""

from __future__ import annotations

import torch

from tfwm.types import Tactile


def normalize_tactile(tactile: Tactile, min_val: float = 0.0, max_val: float = 1.0) -> Tactile:
    """Normalize tactile readings to [0, 1]."""
    return torch.clamp((tactile - min_val) / (max_val - min_val), 0.0, 1.0)


def tactile_gradient(tactile: Tactile) -> Tactile:
    """Compute spatial gradient magnitude."""
    dx = torch.zeros_like(tactile)
    dy = torch.zeros_like(tactile)

    dx[:, 1:-1] = (tactile[:, 2:] - tactile[:, :-2]) / 2.0
    dx[:, 0] = tactile[:, 1] - tactile[:, 0]
    dx[:, -1] = tactile[:, -1] - tactile[:, -2]

    dy[1:-1, :] = (tactile[2:, :] - tactile[:-2, :]) / 2.0
    dy[0, :] = tactile[1, :] - tactile[0, :]
    dy[-1, :] = tactile[-1, :] - tactile[-2, :]

    return torch.sqrt(dx**2 + dy**2)


def tactile_curvature(tactile: Tactile) -> Tactile:
    """Compute approximate curvature of tactile surface."""
    laplacian = (
        torch.roll(tactile, shifts=1, dims=0)
        + torch.roll(tactile, shifts=-1, dims=0)
        + torch.roll(tactile, shifts=1, dims=1)
        + torch.roll(tactile, shifts=-1, dims=1)
        - 4 * tactile
    )
    return laplacian


def tactile_contact_mask(tactile: Tactile, threshold: float = 0.1) -> torch.Tensor:
    """Create binary contact mask from tactile readings."""
    return (tactile > threshold).to(torch.float32)


def tactile_centroid(tactile: Tactile) -> tuple[float, float]:
    """Compute centroid of contact region."""
    h, w = tactile.shape
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=tactile.dtype, device=tactile.device),
        torch.arange(w, dtype=tactile.dtype, device=tactile.device),
        indexing="ij",
    )
    total_mass = tactile.sum()
    if total_mass == 0:
        return float(h / 2), float(w / 2)

    cy = (y_coords * tactile).sum() / total_mass
    cx = (x_coords * tactile).sum() / total_mass
    return float(cy), float(cx)
