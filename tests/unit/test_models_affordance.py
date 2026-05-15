"""Tests for affordance models."""

from __future__ import annotations

import torch
from tfwm.models.affordance import ActionPredictor, AffordancePredictor, TactileGatingNetwork


class TestAffordanceModels:
    def test_affordance_predictor(self):
        model = AffordancePredictor(n_affordances=10, d_latent=64)
        tactile_latent = torch.ones((64,), dtype=torch.float32)
        logits = model(tactile_latent)
        assert logits.shape == (1, 10)

    def test_affordance_predictor_batch(self):
        model = AffordancePredictor(n_affordances=10, d_latent=64)
        tactile_latent = torch.ones((4, 64), dtype=torch.float32)
        logits = model(tactile_latent)
        assert logits.shape == (4, 10)

    def test_tactile_gating_network(self):
        model = TactileGatingNetwork(n_sensors=2, d_latent=64, d_proprio=12)
        tactile_latents = torch.ones((2, 64), dtype=torch.float32)
        proprio = torch.ones((12,), dtype=torch.float32)
        gate_scores = model(tactile_latents, proprio)
        assert gate_scores.shape == (1, 2)

    def test_action_predictor(self):
        model = ActionPredictor(d_latent=64, d_proprio=12, d_action=6)
        tactile_latent = torch.ones((64,), dtype=torch.float32)
        proprio = torch.ones((12,), dtype=torch.float32)
        action = model(tactile_latent, proprio)
        assert action.shape == (1, 6)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_action_predictor_batch(self):
        model = ActionPredictor(d_latent=64, d_proprio=12, d_action=6)
        tactile_latent = torch.ones((4, 64), dtype=torch.float32)
        proprio = torch.ones((4, 12), dtype=torch.float32)
        action = model(tactile_latent, proprio)
        assert action.shape == (4, 6)
