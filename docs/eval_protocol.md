# Evaluation Protocol

Evaluation should report task success and the cost of achieving it.

## Benchmark Runner

Run the local benchmark pipeline with:

```bash
python scripts/generate_synthetic_dataset.py --episodes 12 --steps 48
python scripts/run_benchmark.py
```

or through Hydra:

```bash
tfwm eval eval.data_dir=data/raw/synthetic
```

Both paths write `benchmark_report.json` and `benchmark_report.md` with
bootstrap confidence intervals and configured claim checks.

## Core Metrics

- success rate;
- average return;
- camera queries per episode;
- slip events per episode;
- force violations per episode and per step;
- latency for planning and inference;
- sim-to-real transfer gap when hardware data is available.

## Baselines

Compare against:

- tactile-only world model;
- vision-only or vision-first policy;
- vision-always multimodal policy;
- heuristic camera-query policy;
- oracle query schedule when available.

## Reporting

For every benchmark run, record:

- config overrides;
- git commit or source snapshot;
- dataset manifest and split files;
- random seeds;
- aggregate metrics with confidence intervals;
- failure cases with episode IDs.

## Claim Gates

The default benchmark claims in `configs/eval/benchmark.yaml` are intentionally
explicit. A result should not be described as supporting the paper claims unless
the corresponding claim checks pass on held-out seeds and baseline comparisons
are present.
