"""Tests for benchmark reporting."""

from __future__ import annotations

import json

from tfwm.eval.benchmark import (
    benchmark_report,
    episode_fingerprint,
    evaluate_claims,
    load_json_episodes,
    render_markdown_report,
    validate_episodes,
    write_report_artifacts,
)


def _episode(success: bool, steps: int, slips: int, queries: int) -> dict[str, object]:
    return {
        "metadata": {
            "success": success,
            "steps": steps,
            "slip_events": slips,
            "camera_queries": queries,
            "force_violations": 0,
        }
    }


def test_benchmark_report_contains_intervals() -> None:
    episodes = [_episode(True, 10, 1, 2), _episode(False, 20, 2, 4)]

    report = benchmark_report(
        episodes,
        metric_names=["success_rate", "camera_query_rate"],
        bootstrap_samples=8,
        seed=3,
    )

    assert report["metadata"]["episodes"] == 2
    assert len(report["metadata"]["fingerprint"]) == 64
    assert report["metrics"]["success_rate"]["mean"] == 0.5
    assert report["metrics"]["camera_query_rate"]["mean"] == 0.2
    assert "ci_low" in report["metrics"]["success_rate"]


def test_claim_checks_and_markdown() -> None:
    episodes = [_episode(True, 10, 0, 1)]
    report = benchmark_report(episodes, metric_names=["success_rate"], bootstrap_samples=0)

    checks = evaluate_claims(
        report["metrics"],
        [{"metric": "success_rate", "operator": ">=", "target": 0.9}],
    )
    markdown = render_markdown_report(report, title="Smoke Benchmark", claim_checks=checks)

    assert checks[0].passed is True
    assert "Smoke Benchmark" in markdown
    assert "`success_rate`" in markdown


def test_load_and_write_report_artifacts(tmp_path) -> None:
    data_dir = tmp_path / "episodes"
    data_dir.mkdir()
    (data_dir / "episode_00000.json").write_text(json.dumps(_episode(True, 10, 0, 1)))

    episodes = load_json_episodes(data_dir)
    report = benchmark_report(episodes, bootstrap_samples=0)
    json_path, markdown_path = write_report_artifacts(report, tmp_path / "reports", title="Report")

    assert len(episodes) == 1
    assert json_path.exists()
    assert markdown_path.exists()


def test_episode_validation_and_fingerprint() -> None:
    episodes = [_episode(True, 10, 0, 1)]

    validate_episodes(episodes)

    assert episode_fingerprint(episodes) == episode_fingerprint(episodes)
