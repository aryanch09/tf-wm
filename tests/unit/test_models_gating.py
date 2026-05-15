"""Tests for gating modules."""

from __future__ import annotations

import torch
from tfwm.models.gating import HybridGate, VarianceGate, VOIGate


def test_variance_gate_shape_and_range() -> None:
    gate = VarianceGate()
    latent = torch.zeros(2, 3, 4)
    score = gate(latent, {"logvar": torch.zeros_like(latent)})
    assert score.shape == (2, 3)
    assert torch.all((score >= 0.0) & (score <= 1.0))


def test_voi_gate_shape_and_range() -> None:
    gate = VOIGate(d_latent=4)
    latent = torch.zeros(2, 3, 4)
    score = gate(latent, {"logvar": torch.zeros_like(latent)})
    assert score.shape == (2, 3)
    assert torch.all((score >= 0.0) & (score <= 1.0))


def test_hybrid_gate_query_decision() -> None:
    gate = HybridGate(d_latent=4, threshold=0.5, variance_weight=1.0)
    latent = torch.zeros(1, 2, 4)
    score = gate(latent, {"logvar": torch.ones_like(latent)})
    assert gate.query_vision(score).shape == (1, 2)
