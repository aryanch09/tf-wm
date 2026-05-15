# Reproducibility

TF-WM aims to make experiments repeatable across simulator runs, training jobs,
and evaluation sweeps.

## Required Records

Every experiment should preserve:

- resolved Hydra config;
- package versions;
- random seeds;
- dataset manifest and split files;
- checkpoint path and checksum;
- evaluation command;
- aggregate metrics and per-episode metrics.

## Local Checks

Run the compact smoke suite before sharing changes:

```bash
python scripts/run_smoke_checks.py
```

For docs:

```bash
mkdocs build --strict
```

For a synthetic data sanity check:

```bash
python scripts/generate_synthetic_dataset.py --episodes 8 --steps 32
python scripts/inspect_dataset.py data/raw/synthetic
python scripts/run_benchmark.py --bootstrap-samples 100
```
