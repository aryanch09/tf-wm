#!/usr/bin/env python3
"""Run the local TF-WM benchmark report pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tfwm.eval.benchmark import (
    benchmark_report,
    evaluate_claims,
    load_json_episodes,
    write_report_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--pattern", default="episode_*.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/local"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_json_episodes(args.data_dir, pattern=args.pattern)
    report = benchmark_report(
        episodes,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    claims = [
        {"metric": "success_rate", "operator": ">=", "target": 0.75},
        {"metric": "camera_query_rate", "operator": "<=", "target": 0.20},
        {"metric": "slip_rate", "operator": "<=", "target": 0.08},
        {"metric": "force_violation_rate", "operator": "<=", "target": 0.01},
    ]
    checks = evaluate_claims(report["metrics"], claims)
    json_path, markdown_path = write_report_artifacts(
        report,
        args.output_dir,
        title="TF-WM Local Benchmark",
        claim_checks=checks,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
