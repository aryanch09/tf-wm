"""Tests for contract-compatible tactile encoders."""

from __future__ import annotations

import torch
from tfwm.models.tactile_encoder import HybridTactileEncoder, TCNTactileEncoder


def test_tcn_tactile_encoder_returns_latent_dist() -> None:
    encoder = TCNTactileEncoder(n_taxels=8, n_channels=2, d_latent=16)
    tactile = torch.ones(3, 5, 8, 2)
    latent = encoder(tactile)
    assert latent["mu"].shape == (3, 5, 16)
    assert latent["logvar"].shape == (3, 5, 16)


def test_hybrid_tactile_encoder_attention_mask() -> None:
    encoder = HybridTactileEncoder(n_taxels=4, n_channels=1, d_latent=8)
    tactile = torch.ones(2, 6, 4, 1)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    latent = encoder(tactile, attention_mask=mask)
    assert latent["mu"].shape == (2, 6, 8)
