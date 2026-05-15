"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import torch
from tfwm.eval.metrics import (
    average_return,
    camera_query_rate,
    force_violation_rate,
    safety_violations,
    slip_rate,
    steps_to_success,
    success_rate,
    tactile_prediction_rmse,
    voi_calibration_ece,
)


def test_episode_metrics() -> None:
    episodes = [
        {
            "metadata": {
                "success": True,
                "steps": 10,
                "slip_events": 1,
                "camera_queries": 2,
                "force_violations": 0,
            },
            "rewards": [1.0, 2.0],
        },
        {
            "metadata": {
                "success": False,
                "steps": 20,
                "slip_events": 2,
                "camera_queries": 4,
                "force_violations": 3,
            },
            "rewards": [0.0],
        },
    ]

    assert success_rate(episodes) == 0.5
    assert steps_to_success(episodes) == 10.0
    assert slip_rate(episodes) == 0.1
    assert camera_query_rate(episodes) == 0.2
    assert force_violation_rate(episodes) == 0.1
    assert average_return(episodes) == 1.5


def test_steps_to_success_no_success() -> None:
    assert math.isinf(steps_to_success([{"success": False, "steps": 4}]))


def test_safety_violations() -> None:
    episodes = [{"forces_N": torch.tensor([0.1, 2.0, 4.0])}, {"forces_N": [5.0]}]
    assert safety_violations(episodes, force_thresh_N=3.0) == 2


def test_tactile_prediction_rmse() -> None:
    rmse = tactile_prediction_rmse(
        [{"pred_tactile": torch.zeros(2, 2), "gt_tactile": torch.ones(2, 2)}]
    )
    assert rmse == 1.0


def test_voi_calibration_ece() -> None:
    ece = voi_calibration_ece(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), n_bins=2)
    assert ece == 0.0
