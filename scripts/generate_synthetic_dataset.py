#!/usr/bin/env python3
"""Generate a small TF-WM-compatible synthetic dataset.

This script is intentionally dependency-light and writes JSON episodes that can be
loaded by ``tfwm.data.DirectoryDataset``. It is useful for smoke tests, tutorials,
and checking data pipeline changes before a simulator is available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_episode(
    episode_id: int,
    *,
    steps: int,
    n_taxels: int,
    action_dim: int,
    seed: int,
    task_name: str,
    sim_name: str,
) -> dict[str, object]:
    rng = random.Random(seed + episode_id)
    observations = []
    actions = []
    rewards = []
    dones = []
    slip_events = 0
    camera_queries = 0
    late_contact = []

    for step in range(steps):
        timestamp = step / 50.0
        contact_ramp = min(1.0, max(0.0, (step - steps * 0.2) / max(1.0, steps * 0.5)))
        tactile = [
            round(max(0.0, min(1.0, rng.gauss(contact_ramp * 0.35, 0.06))), 6)
            for _ in range(n_taxels)
        ]
        contact_events = [int(value > 0.15) for value in tactile]
        proprio = [
            round(rng.uniform(-0.5, 0.5) + 0.01 * step, 6),
            round(rng.uniform(-0.5, 0.5), 6),
            round(rng.uniform(-0.25, 0.25), 6),
            round(contact_ramp, 6),
            round(sum(tactile) / n_taxels, 6),
            round(max(tactile), 6),
        ]
        action = [round(rng.uniform(-1.0, 1.0), 6) for _ in range(action_dim)]
        slip = int(step > 0 and rng.random() < 0.02 + 0.04 * contact_ramp)
        slip_events += slip
        should_query_camera = int(step % 10 == 0 or (slip and rng.random() < 0.5))
        camera_queries += should_query_camera
        if step >= int(steps * 0.6):
            late_contact.append(max(tactile))

        observations.append(
            {
                "tactile": tactile,
                "vision": None,
                "proprio": proprio,
                "contact_events": contact_events,
                "timestamp": timestamp,
            }
        )
        actions.append({"action": action, "timestamp": timestamp})
        rewards.append(round(max(tactile) - 0.05 * slip, 6))
        dones.append(step == steps - 1)

    mean_late_contact = sum(late_contact) / max(1, len(late_contact))
    success = mean_late_contact > 0.24 and slip_events < max(2, int(steps * 0.12))
    return {
        "metadata": {
            "task_name": task_name,
            "sim_name": sim_name,
            "seed": seed + episode_id,
            "randomization_seed": seed * 1000 + episode_id,
            "success": success,
            "steps": steps,
            "duration_s": round(steps / 50.0, 6),
            "camera_queries": camera_queries,
            "slip_events": slip_events,
            "force_violations": 0,
        },
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def _write_manifest(output_dir: Path, records: list[dict[str, object]]) -> None:
    manifest = output_dir / "manifest.csv"
    header = [
        "episode_id",
        "path",
        "split",
        "task_name",
        "sim_name",
        "seed",
        "steps",
        "success",
        "camera_queries",
        "slip_events",
    ]
    lines = [",".join(header)]
    for record in records:
        lines.append(",".join(str(record[key]) for key in header))
    manifest.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=_repo_root() / "data" / "raw" / "synthetic"
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n-taxels", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-name", default="inhand_reorient")
    parser.add_argument("--sim-name", default="synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for episode_id in range(args.episodes):
        episode = _build_episode(
            episode_id,
            steps=args.steps,
            n_taxels=args.n_taxels,
            action_dim=args.action_dim,
            seed=args.seed,
            task_name=args.task_name,
            sim_name=args.sim_name,
        )
        split = "train" if episode_id < int(args.episodes * 0.8) else "val"
        path = args.output_dir / f"episode_{episode_id:05d}.json"
        path.write_text(json.dumps(episode, indent=2) + "\n")
        metadata = episode["metadata"]
        assert isinstance(metadata, dict)
        records.append(
            {
                "episode_id": episode_id,
                "path": path.relative_to(_repo_root()),
                "split": split,
                "task_name": metadata["task_name"],
                "sim_name": metadata["sim_name"],
                "seed": metadata["seed"],
                "steps": metadata["steps"],
                "success": metadata["success"],
                "camera_queries": metadata["camera_queries"],
                "slip_events": metadata["slip_events"],
            }
        )

    _write_manifest(args.output_dir, records)
    print(f"Wrote {args.episodes} episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
