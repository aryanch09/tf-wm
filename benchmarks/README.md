# Benchmarks

This directory defines the public evaluation surface for TF-WM. Benchmarks are
designed to measure both manipulation quality and the cost of sensing.

## Current Benchmark: In-Hand Reorientation Under Occlusion

Primary metrics:

- `success_rate`: fraction of episodes reaching the task goal.
- `steps_to_success`: successful episode length, lower is better when success
  is held constant.
- `camera_query_rate`: camera queries per environment step.
- `slip_rate`: detected slip events per environment step.

Default claim targets are configured in `configs/eval/benchmark.yaml`.

## Local Smoke Benchmark

```bash
python scripts/generate_synthetic_dataset.py --episodes 12 --steps 48
python scripts/run_benchmark.py
```

Reports are written to `outputs/benchmarks/local` as JSON and Markdown.
