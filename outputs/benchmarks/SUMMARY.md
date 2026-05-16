# TF-WM Camera-Ready Benchmark Sweep

Per-task summaries written by `scripts/run_camera_ready_sweep.py`. Each
sub-folder contains `benchmark_report.json` (machine-readable) and
`benchmark_report.md` (human-readable) with claim checks.

## Multi-task results (TF-WM, ours)

30 matched-seed episodes per task, 1000 bootstrap samples, task-specific
seeds.

| Task                              | Success $\uparrow$ | Q-rate $\downarrow$ | Slip $\downarrow$ | F-viol $\downarrow$ |
|-----------------------------------|:------------------:|:--------------------:|:------------------:|:--------------------:|
| `inhand_reorient` (headline)      | **1.000** [1.00,1.00] | **0.111** [.106,.118] | 0.028 [.013,.043] | **0.000** |
| `peg_in_hole`                     | 0.967 [.900,1.00]     | 0.120 [.115,.127]     | 0.037 [.028,.047] | **0.000** |
| `blind_retrieve`                  | **1.000** [1.00,1.00] | 0.115 [.112,.119]     | 0.033 [.026,.041] | **0.000** |
| `tool_use`                        | 0.967 [.900,1.00]     | 0.120 [.115,.125]     | 0.040 [.030,.050] | **0.000** |

**All four claim gates pass on all four tasks.** Force-limit violations are
exactly zero across every evaluated episode.

## Reproducing this sweep

```bash
python scripts/run_camera_ready_sweep.py --episodes 30 --bootstrap 1000
```

Per-task fingerprints are stamped in each `benchmark_report.json` under
`metadata.fingerprint`.

## Baseline sweep (in-progress)

The four card-defined baselines (`tactile_only`, `vision_always`,
`heuristic_gate`, `oracle_schedule`) are wired through the same evaluator
via Hydra overrides; the orchestration entrypoint is:

```bash
python scripts/run_baseline_sweep.py --tasks all --baselines all
```

Each `(task, baseline)` cell produces its own
`outputs/benchmarks/<task>__<baseline>/benchmark_report.{json,md}` so the
camera-ready head-to-head table can be assembled with one `glob`.
