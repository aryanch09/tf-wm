#!/usr/bin/env python3
"""Camera-ready baseline sweep: TF-WM vs four card-defined baselines.

Wraps the Hydra training entrypoint to produce one evaluation report per
(task, baseline) cell. Each cell is configured by overriding the gate and
the encoder/dynamics so the baseline matches its card-defined design:

  * tactile_only        : gate=none,        encoder=hybrid,  dynamics=rssm
  * vision_always       : gate=always,      encoder=hybrid,  dynamics=rssm
  * heuristic_gate      : gate=variance,    encoder=hybrid,  dynamics=rssm
  * voi_gate (TF-WM)    : gate=voi,         encoder=hybrid,  dynamics=rssm
  * oracle_schedule     : gate=oracle,      encoder=hybrid,  dynamics=rssm

Real evaluation requires a trained world model and the simulator; this script
is the orchestration layer the camera-ready will use. It writes one
``benchmark_report.{json,md}`` per cell under
``outputs/benchmarks/<task>__<baseline>/`` so downstream notebooks can build
the head-to-head table in one ``glob``.

Usage:
    python scripts/run_baseline_sweep.py --tasks all --baselines all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TASKS = ["inhand_reorient", "peg_in_hole", "blind_retrieve", "tool_use"]

BASELINES = {
    # name              : Hydra override
    "tactile_only"      : "model.gating=none",
    "vision_always"     : "model.gating=always",
    "heuristic_gate"    : "model.gating=variance",
    "voi_gate"          : "model.gating=voi",   # ours
    "oracle_schedule"   : "model.gating=oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--baselines", nargs="+", default=["all"])
    parser.add_argument("--seeds", type=int, default=30,
                        help="Episodes per (task, baseline) cell.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned commands, don't execute.")
    return parser.parse_args()


def _expand(arg: list[str], universe: list[str]) -> list[str]:
    if arg == ["all"]:
        return universe
    return arg


def main() -> None:
    args = parse_args()
    tasks = _expand(args.tasks, TASKS)
    baselines = _expand(args.baselines, list(BASELINES))

    root = Path(__file__).resolve().parents[1]
    for task in tasks:
        for bl in baselines:
            override = BASELINES[bl]
            out_dir = root / "outputs" / "benchmarks" / f"{task}__{bl}"
            cmd = [
                "tfwm", "eval",
                f"task={task}",
                f"eval.benchmark.bootstrap_samples={args.bootstrap}",
                f"eval.benchmark.episode_count={args.seeds}",
                override,
                f"hydra.run.dir={out_dir}",
            ]
            print(f"[run] task={task:<18s} baseline={bl:<16s} -> {out_dir}")
            print("      " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
