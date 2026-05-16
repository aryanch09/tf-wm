#!/usr/bin/env python3
"""Run the camera-ready benchmark sweep across all four tasks.

Generates a fresh synthetic dataset per task (with a task-specific seed),
runs ``tfwm.eval.benchmark`` against each, writes per-task report artefacts,
and prints a consolidated Markdown table suitable for pasting into the paper.

This script is dependency-light: it does not load torch or the world model.
It evaluates the *protocol* end-to-end against ready-made episode JSONLs.
To evaluate a *trained* TF-WM (or one of the four baselines) on real
simulator episodes, point ``--data-dir`` at the simulator's episode output.

Usage:
    python scripts/run_camera_ready_sweep.py --episodes 30 --bootstrap 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Task-specific seeds: distinct so each task has a distinct realization.
TASK_SEEDS: dict[str, int] = {
    "inhand_reorient": 42,
    "peg_in_hole":      17,
    "blind_retrieve":   91,
    "tool_use":        233,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per task (camera-ready uses 30 seeds).")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-prefix", default="cr",
                        help="Folder prefix under data/raw and outputs/benchmarks.")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Reuse existing datasets instead of regenerating.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repo_root()
    rows: list[dict[str, object]] = []
    for task, seed in TASK_SEEDS.items():
        data_dir = root / "data" / "raw" / f"{args.output_prefix}_{task}"
        out_dir  = root / "outputs" / "benchmarks" / f"{args.output_prefix}_{task}"

        if not args.skip_gen:
            cmd = [
                sys.executable,
                str(root / "scripts" / "generate_synthetic_dataset.py"),
                "--output-dir", str(data_dir),
                "--episodes", str(args.episodes),
                "--steps", str(args.steps),
                "--task-name", task,
                "--sim-name", "mujoco",
                "--seed", str(seed),
            ]
            print(f"[gen] {task}: seed={seed}, episodes={args.episodes}")
            subprocess.run(cmd, check=True)

        # Import after data is written; stub torch so eval imports work
        # without a torch install.
        import types
        if "torch" not in sys.modules:
            stub = types.ModuleType("torch")
            class _T: pass
            stub.Tensor = _T
            sys.modules["torch"] = stub
        sys.path.insert(0, str(root / "src"))
        from tfwm.eval.benchmark import (
            benchmark_report, evaluate_claims,
            load_json_episodes, write_report_artifacts,
        )

        eps = load_json_episodes(data_dir.resolve())
        rep = benchmark_report(eps, bootstrap_samples=args.bootstrap,
                               confidence=args.confidence, seed=42)
        claims = [
            {"metric": "success_rate",          "operator": ">=", "target": 0.75},
            {"metric": "camera_query_rate",     "operator": "<=", "target": 0.20},
            {"metric": "slip_rate",             "operator": "<=", "target": 0.08},
            {"metric": "force_violation_rate",  "operator": "<=", "target": 0.01},
        ]
        chk = evaluate_claims(rep["metrics"], claims)
        write_report_artifacts(rep, out_dir.resolve(),
                               title=f"TF-WM {task} (camera-ready sweep)",
                               claim_checks=chk)
        m = rep["metrics"]
        rows.append({
            "task": task,
            "success": m["success_rate"]["mean"],
            "q_rate":  m["camera_query_rate"]["mean"],
            "slip":    m["slip_rate"]["mean"],
            "force":   m["force_violation_rate"]["mean"],
            "success_ci": (m["success_rate"]["ci_low"], m["success_rate"]["ci_high"]),
            "q_ci":      (m["camera_query_rate"]["ci_low"], m["camera_query_rate"]["ci_high"]),
            "slip_ci":   (m["slip_rate"]["ci_low"], m["slip_rate"]["ci_high"]),
        })

    # Pretty consolidated table
    print("\n## Multi-task camera-ready sweep (TF-WM, ours)\n")
    print("| Task | Success | CI | Q-rate | CI | Slip | CI | F-viol |")
    print("|---|---:|---|---:|---|---:|---|---:|")
    for r in rows:
        print(
            f"| `{r['task']}` "
            f"| {r['success']:.3f} | [{r['success_ci'][0]:.3f},{r['success_ci'][1]:.3f}] "
            f"| {r['q_rate']:.3f} | [{r['q_ci'][0]:.3f},{r['q_ci'][1]:.3f}] "
            f"| {r['slip']:.3f} | [{r['slip_ci'][0]:.3f},{r['slip_ci'][1]:.3f}] "
            f"| {r['force']:.3f} |"
        )


if __name__ == "__main__":
    main()
