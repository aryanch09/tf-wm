#!/usr/bin/env python3
"""Inspect TF-WM JSON episodes and report dataset health metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing episode_*.json files")
    parser.add_argument("--pattern", default="*.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No episodes found in {args.data_dir} matching {args.pattern}")

    steps = []
    rewards = []
    successes = 0
    camera_queries = 0
    slip_events = 0
    tasks: set[str] = set()
    sims: set[str] = set()

    for path in files:
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        tasks.add(str(metadata["task_name"]))
        sims.add(str(metadata["sim_name"]))
        steps.append(int(metadata["steps"]))
        successes += int(bool(metadata["success"]))
        camera_queries += int(metadata["camera_queries"])
        slip_events += int(metadata["slip_events"])
        rewards.extend(float(value) for value in data["rewards"])

    print(f"Episodes: {len(files)}")
    print(f"Tasks: {', '.join(sorted(tasks))}")
    print(f"Simulators: {', '.join(sorted(sims))}")
    print(f"Steps: min={min(steps)} mean={mean(steps):.1f} max={max(steps)}")
    print(f"Success rate: {successes / len(files):.2%}")
    print(f"Camera queries / episode: {camera_queries / len(files):.1f}")
    print(f"Slip events / episode: {slip_events / len(files):.1f}")
    print(f"Reward: min={min(rewards):.3f} mean={mean(rewards):.3f} max={max(rewards):.3f}")


if __name__ == "__main__":
    main()
