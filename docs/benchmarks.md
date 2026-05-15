# Benchmarks

Benchmarks are the public contract for TF-WM claims. They combine task success,
sensor cost, and safety diagnostics so a tactile-first method cannot hide poor
behavior behind a single aggregate score.

## In-Hand Reorientation Under Occlusion

Configuration:

- benchmark card: `benchmarks/inhand_reorient_occluded.yaml`
- Hydra config: `configs/eval/benchmark.yaml`
- local runner: `scripts/run_benchmark.py`

Primary questions:

- Does tactile-first prediction preserve task success under occlusion?
- Does VOI gating reduce camera use without a large success penalty?
- Do slip and force diagnostics stay within deployment limits?

Reported metrics:

- `success_rate`: fraction of episodes reaching the task goal.
- `steps_to_success`: mean length of successful episodes.
- `average_return`: mean episodic reward.
- `camera_query_rate`: camera queries per environment step.
- `slip_rate`: detected slip events per environment step.
- `force_violation_rate`: force-limit events per environment step.

## Report Artifacts

Each run writes:

- `benchmark_report.json`: machine-readable metrics, confidence intervals, and
  claim checks.
- `benchmark_report.md`: compact human-readable summary for lab notes,
  reviews, and paper tables.

## Baseline Expectations

Conference-ready results should include at least:

- tactile-only world model;
- vision-always multimodal policy;
- heuristic uncertainty gate;
- TF-WM VOI gate;
- oracle or upper-bound query schedule when available.

Any reported improvement should include matched seeds, confidence intervals, and
the exact dataset manifest.
