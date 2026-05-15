"""Tests for dynamics models."""

from __future__ import annotations

import torch
from tfwm.models.dynamics import DynamicsModel, PriorDynamicsModel


class TestDynamicsModels:
    def test_dynamics_model(self):
        model = DynamicsModel(d_latent=64, d_action=6, d_proprio=12)
        tactile_latent = torch.ones((64,), dtype=torch.float32)
        proprio = torch.ones((12,), dtype=torch.float32)
        action = torch.ones((6,), dtype=torch.float32)
        dist = model(tactile_latent, proprio, action)
        assert "mu" in dist and "logvar" in dist
        assert dist["mu"].shape == (1, 64)
        assert dist["logvar"].shape == (1, 64)

    def test_dynamics_model_batch(self):
        model = DynamicsModel(d_latent=64, d_action=6, d_proprio=12)
        tactile_latent = torch.ones((4, 64), dtype=torch.float32)
        proprio = torch.ones((4, 12), dtype=torch.float32)
        action = torch.ones((4, 6), dtype=torch.float32)
        dist = model(tactile_latent, proprio, action)
        assert dist["mu"].shape == (4, 64)
        assert dist["logvar"].shape == (4, 64)

    def test_prior_dynamics_model(self):
        model = PriorDynamicsModel(d_latent=64)
        prev_latent = torch.ones((64,), dtype=torch.float32)
        dist = model(prev_latent)
        assert dist["mu"].shape == (1, 64)
        assert dist["logvar"].shape == (1, 64)

    def test_prior_dynamics_model_batch(self):
        model = PriorDynamicsModel(d_latent=64)
        prev_latent = torch.ones((4, 64), dtype=torch.float32)
        dist = model(prev_latent)
        assert dist["mu"].shape == (4, 64)
        assert dist["logvar"].shape == (4, 64)
