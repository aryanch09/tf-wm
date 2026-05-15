"""Evaluation tools for TF-WM."""

from .benchmark import (
    benchmark_report,
    bootstrap_interval,
    episode_fingerprint,
    evaluate_claims,
    load_json_episodes,
    render_markdown_report,
    summarize_episodes,
    validate_episodes,
    write_report_artifacts,
)
from .evaluator import Evaluator
from .metrics import (
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

__all__ = [
    "Evaluator",
    "average_return",
    "benchmark_report",
    "bootstrap_interval",
    "camera_query_rate",
    "episode_fingerprint",
    "evaluate_claims",
    "force_violation_rate",
    "load_json_episodes",
    "render_markdown_report",
    "safety_violations",
    "slip_rate",
    "steps_to_success",
    "summarize_episodes",
    "success_rate",
    "tactile_prediction_rmse",
    "validate_episodes",
    "voi_calibration_ece",
    "write_report_artifacts",
]
