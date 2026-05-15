# Contributing

TF-WM is organized as a reproducible research codebase. Contributions should
improve one of three things: scientific validity, implementation reliability, or
clarity for future researchers.

## Local Checks

```bash
python scripts/generate_synthetic_dataset.py --episodes 12 --steps 48
python scripts/run_benchmark.py --bootstrap-samples 100
python scripts/run_smoke_checks.py
mkdocs build --strict
```

## Expectations

- Add or update tests for public behavior.
- Keep benchmark metrics reproducible from committed configs.
- Do not commit large datasets, model checkpoints, secrets, or private robot
  logs.
- Record non-obvious experiment settings in configs or docs.
- Prefer small, reviewable changes over sweeping rewrites.

## Research Claims

Claims should be treated as hypotheses until benchmark reports support them.
Every result table should cite the dataset manifest, split files, seeds, config,
and generated report artifact.
