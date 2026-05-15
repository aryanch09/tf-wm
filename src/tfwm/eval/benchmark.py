"""Benchmark reporting utilities for TF-WM experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tfwm.eval.metrics import (
    average_return,
    camera_query_rate,
    force_violation_rate,
    slip_rate,
    steps_to_success,
    success_rate,
)

EpisodeRecord = Mapping[str, Any]
MetricFn = Callable[[Sequence[EpisodeRecord]], float]


@dataclass(frozen=True)
class ConfidenceInterval:
    """Bootstrap confidence interval for a scalar metric."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class ClaimCheck:
    """Result of checking a benchmark metric against a target."""

    metric: str
    operator: str
    target: float
    value: float
    passed: bool


METRICS: dict[str, MetricFn] = {
    "success_rate": success_rate,
    "steps_to_success": steps_to_success,
    "average_return": average_return,
    "slip_rate": slip_rate,
    "camera_query_rate": camera_query_rate,
    "force_violation_rate": force_violation_rate,
}


def load_json_episodes(data_dir: Path, pattern: str = "episode_*.json") -> list[EpisodeRecord]:
    """Load JSON episode records from a directory."""

    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No benchmark episodes found in {data_dir} matching {pattern}")
    episodes = [json.loads(path.read_text()) for path in files]
    validate_episodes(episodes)
    return episodes


def validate_episodes(episodes: Sequence[EpisodeRecord]) -> None:
    """Validate the minimal benchmark episode contract."""

    for index, episode in enumerate(episodes):
        metadata = episode.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Episode {index} is missing mapping metadata")
        for key in ("success", "steps", "camera_queries", "slip_events", "force_violations"):
            if key not in metadata:
                raise ValueError(f"Episode {index} metadata is missing {key!r}")
        steps = int(metadata["steps"])
        if steps <= 0:
            raise ValueError(f"Episode {index} has non-positive steps: {steps}")
        if "rewards" in episode and len(episode["rewards"]) != steps:
            raise ValueError(f"Episode {index} rewards length does not match steps")


def episode_fingerprint(episodes: Sequence[EpisodeRecord]) -> str:
    """Return a stable content hash for a benchmark episode set."""

    payload = json.dumps(episodes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_episodes(episodes: Sequence[EpisodeRecord]) -> dict[str, float]:
    """Compute the standard benchmark metric set for a collection of episodes."""

    return {name: fn(episodes) for name, fn in METRICS.items()}


def bootstrap_interval(
    episodes: Sequence[EpisodeRecord],
    metric_fn: MetricFn,
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Estimate a nonparametric bootstrap confidence interval."""

    if not episodes:
        return ConfidenceInterval(mean=0.0, low=0.0, high=0.0)
    if samples <= 0:
        value = float(metric_fn(episodes))
        return ConfidenceInterval(mean=value, low=value, high=value)

    rng = np.random.default_rng(seed)
    values = []
    n = len(episodes)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        sample = [episodes[int(index)] for index in indices]
        value = float(metric_fn(sample))
        if np.isfinite(value):
            values.append(value)

    mean = float(metric_fn(episodes))
    if not values:
        return ConfidenceInterval(mean=mean, low=mean, high=mean)

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [alpha, 1.0 - alpha])
    return ConfidenceInterval(mean=mean, low=float(low), high=float(high))


def benchmark_report(
    episodes: Sequence[EpisodeRecord],
    *,
    metric_names: Sequence[str] | None = None,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a benchmark report with scalar metrics and confidence intervals."""

    selected = list(metric_names or METRICS.keys())
    unknown = sorted(set(selected) - set(METRICS))
    if unknown:
        raise ValueError(f"Unknown benchmark metrics: {', '.join(unknown)}")

    metrics = {}
    for index, name in enumerate(selected):
        interval = bootstrap_interval(
            episodes,
            METRICS[name],
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed + index,
        )
        metrics[name] = {
            "mean": interval.mean,
            "ci_low": interval.low,
            "ci_high": interval.high,
        }

    metadata = {
        "episodes": len(episodes),
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "fingerprint": episode_fingerprint(episodes),
    }
    return {"metadata": metadata, "metrics": metrics}


def evaluate_claims(
    metrics: Mapping[str, Mapping[str, float]],
    claims: Sequence[Mapping[str, Any]],
) -> list[ClaimCheck]:
    """Check configured research claims against benchmark metric means."""

    checks = []
    for claim in claims:
        metric = str(claim["metric"])
        if metric not in metrics:
            raise ValueError(f"Claim references unknown metric: {metric}")
        operator = str(claim.get("operator", ">="))
        target = float(claim["target"])
        value = float(metrics[metric]["mean"])
        if operator == ">=":
            passed = value >= target
        elif operator == "<=":
            passed = value <= target
        elif operator == ">":
            passed = value > target
        elif operator == "<":
            passed = value < target
        else:
            raise ValueError(f"Unsupported claim operator: {operator}")
        checks.append(ClaimCheck(metric, operator, target, value, passed))
    return checks


def render_markdown_report(
    report: Mapping[str, Any],
    *,
    title: str,
    claim_checks: Sequence[ClaimCheck] = (),
) -> str:
    """Render a compact Markdown benchmark report."""

    lines = [
        f"# {title}",
        "",
        f"Episodes: {report['metadata']['episodes']}",
        f"Confidence: {report['metadata']['confidence']:.0%}",
        f"Bootstrap samples: {report['metadata']['bootstrap_samples']}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean | CI Low | CI High |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in report["metrics"].items():
        lines.append(
            f"| `{name}` | {values['mean']:.4f} | {values['ci_low']:.4f} | {values['ci_high']:.4f} |"
        )

    if claim_checks:
        lines.extend(
            [
                "",
                "## Claim Checks",
                "",
                "| Metric | Target | Value | Result |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for check in claim_checks:
            result = "pass" if check.passed else "fail"
            lines.append(
                f"| `{check.metric}` | {check.operator} {check.target:.4f} | "
                f"{check.value:.4f} | {result} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report_artifacts(
    report: Mapping[str, Any],
    output_dir: Path,
    *,
    title: str,
    claim_checks: Sequence[ClaimCheck] = (),
) -> tuple[Path, Path]:
    """Write JSON and Markdown benchmark reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_report.json"
    markdown_path = output_dir / "benchmark_report.md"
    serializable = dict(report)
    if claim_checks:
        serializable["claim_checks"] = [
            {
                "metric": check.metric,
                "operator": check.operator,
                "target": check.target,
                "value": check.value,
                "passed": check.passed,
            }
            for check in claim_checks
        ]
    json_path.write_text(json.dumps(serializable, indent=2) + "\n")
    markdown_path.write_text(render_markdown_report(report, title=title, claim_checks=claim_checks))
    return json_path, markdown_path
