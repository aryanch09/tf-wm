"""Evaluation metrics for TF-WM experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch

EpisodeLike = Mapping[str, object]


def _metadata_value(episode: EpisodeLike, name: str, default: object = 0) -> object:
    metadata = episode.get("metadata", {})
    if isinstance(metadata, Mapping) and name in metadata:
        return metadata[name]
    return episode.get(name, default)


def success_rate(episodes: Sequence[EpisodeLike]) -> float:
    """Return the fraction of successful episodes."""

    if not episodes:
        return 0.0
    return float(np.mean([bool(_metadata_value(ep, "success", False)) for ep in episodes]))


def steps_to_success(episodes: Sequence[EpisodeLike]) -> float:
    """Return mean successful episode length, or ``inf`` if none succeeded."""

    steps = [
        float(_metadata_value(ep, "steps", 0))
        for ep in episodes
        if bool(_metadata_value(ep, "success", False))
    ]
    return float(np.mean(steps)) if steps else float("inf")


def safety_violations(episodes: Sequence[EpisodeLike], force_thresh_N: float) -> int:
    """Count force samples above ``force_thresh_N`` across episodes."""

    count = 0
    for ep in episodes:
        forces = ep.get("forces_N", [])
        if isinstance(forces, torch.Tensor):
            count += int((forces > force_thresh_N).sum().item())
        else:
            count += int(np.sum(np.asarray(forces, dtype=np.float32) > force_thresh_N))
    return count


def slip_rate(episodes: Sequence[EpisodeLike]) -> float:
    """Return slips per step across episodes."""

    slips = sum(float(_metadata_value(ep, "slip_events", 0)) for ep in episodes)
    steps = sum(float(_metadata_value(ep, "steps", 0)) for ep in episodes)
    return float(slips / max(steps, 1.0))


def camera_query_rate(episodes: Sequence[EpisodeLike]) -> float:
    """Return camera queries per step across episodes."""

    queries = sum(float(_metadata_value(ep, "camera_queries", 0)) for ep in episodes)
    steps = sum(float(_metadata_value(ep, "steps", 0)) for ep in episodes)
    return float(queries / max(steps, 1.0))


def average_return(episodes: Sequence[EpisodeLike]) -> float:
    """Return mean episodic return."""

    if not episodes:
        return 0.0
    returns = []
    for ep in episodes:
        rewards = ep.get("rewards", [])
        returns.append(float(np.sum(np.asarray(rewards, dtype=np.float32))))
    return float(np.mean(returns))


def force_violation_rate(episodes: Sequence[EpisodeLike]) -> float:
    """Return force violations per step across episodes."""

    violations = sum(float(_metadata_value(ep, "force_violations", 0)) for ep in episodes)
    steps = sum(float(_metadata_value(ep, "steps", 0)) for ep in episodes)
    return float(violations / max(steps, 1.0))


def tactile_prediction_rmse(rollouts: Iterable[Mapping[str, torch.Tensor]]) -> float:
    """Return RMSE between predicted and ground-truth tactile rollouts."""

    squared_errors = []
    for rollout in rollouts:
        pred = rollout["pred_tactile"].detach().float()
        gt = rollout["gt_tactile"].detach().float()
        squared_errors.append(torch.mean((pred - gt) ** 2))
    if not squared_errors:
        return 0.0
    return float(torch.sqrt(torch.stack(squared_errors).mean()).item())


def voi_calibration_ece(
    predicted_voi: torch.Tensor,
    actual_voi: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected calibration error for VOI predictions."""

    pred = predicted_voi.detach().float().flatten().clamp(0.0, 1.0)
    actual = actual_voi.detach().float().flatten().clamp(0.0, 1.0)
    if pred.numel() != actual.numel():
        raise ValueError("predicted_voi and actual_voi must contain the same number of values")
    if pred.numel() == 0:
        return 0.0

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=pred.device)
    ece = torch.zeros((), device=pred.device)
    for i in range(n_bins):
        lower = bins[i]
        upper = bins[i + 1]
        in_bin = (pred >= lower) & (pred <= upper if i == n_bins - 1 else pred < upper)
        if in_bin.any():
            weight = in_bin.float().mean()
            ece = ece + weight * torch.abs(pred[in_bin].mean() - actual[in_bin].mean())
    return float(ece.item())
